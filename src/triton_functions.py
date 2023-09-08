import triton 
import triton.language as tl
import torch

@triton.jit
def get_freq_multi_tokens(offs_cn, starting_idx, theta: tl.constexpr, NB_TOKENS: tl.constexpr):
    DIM: tl.constexpr = 128  # in model, dim = self.params.dim // self.params.n_heads
    freqs = offs_cn % DIM
    freqs = freqs.to(tl.float32) / DIM
    freqs = tl.math.pow(theta, freqs)
    freqs = (tl.arange(0, NB_TOKENS) + starting_idx)[:, None] / freqs[None, :]
    return tl.cos(freqs), tl.sin(freqs)

@triton.jit
def rbe_triton(x_ptr, out_ptr,
               M, K,
               stride_x_batch, stride_x_m, stride_x_n,
               stride_out_batch, stride_out_m, stride_out_n,
               start_token_position,
               THETA: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    
    pid_batch = tl.program_id(axis=0)
    pid = tl.program_id(axis=1)
    pid_m = pid // tl.cdiv(K, BLOCK_SIZE_K)
    pid_n = pid % tl.cdiv(K, BLOCK_SIZE_K)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K // 2) * 2  # take only even numbers
    x_ptrs = x_ptr + (pid_batch * stride_x_batch + stride_x_m * offs_m[:, None] + stride_x_n * offs_n[None, :])
    x_real_mask = (offs_m[:, None] < M) & (offs_n[None, :] < K)
    real = tl.load(x_ptrs, mask=x_real_mask, other=0.0)
    x_imag_mask = (offs_m[:, None] < M) & (1 + offs_n[None, :] < K)
    imag = tl.load(x_ptrs + 1, mask=x_imag_mask, other=0.0)
    tl.debug_barrier()
    start_block = start_token_position + pid_m * BLOCK_SIZE_M
    cos, sin = get_freq_multi_tokens(offs_cn=offs_n, starting_idx=start_block, theta=THETA, NB_TOKENS=BLOCK_SIZE_M)

    out_real = real * cos - imag * sin
    out_imag = real * sin + imag * cos
    tl.debug_barrier()
    out_ptrs = out_ptr + (
            pid_batch * stride_out_batch + stride_out_m * offs_m[:, None] + stride_out_n * offs_n[None, :])
    out_real_mask = (offs_m[:, None] < M) & (offs_n[None, :] < K)
    tl.store(out_ptrs, out_real, mask=out_real_mask)
    out_imag_mask = (offs_m[:, None] < M) & (1 + offs_n[None, :] < K)
    tl.store(out_ptrs + 1, out_imag, mask=out_imag_mask)


def rbe_triton_wrapper(x: torch.Tensor, pos: int = 0) -> torch.Tensor:
    batch, M, K = x.shape
    out = torch.empty_like(x)
    grid = lambda META: (
        batch, triton.cdiv(META["M"], META["BLOCK_SIZE_M"]) * triton.cdiv(META["K"], META["BLOCK_SIZE_K"]),)

    rbe_triton[grid](x, out,
                     M, K,
                     *x.stride(),
                     *out.stride(),
                     start_token_position=pos, THETA=10000., BLOCK_SIZE_M=2, BLOCK_SIZE_K=1024)
    return out


@triton.jit
def rmsnorm_triton(x_ptr, rms_w_ptr, output_ptr,
                  stride_x_batch, stride_x_m, stride_x_k,
                  stride_rms_w,
                  stride_out_batch, stride_out_m, stride_out_k,
                  N_SIZE: tl.constexpr, eps: tl.constexpr, BLOCK_N_SIZE: tl.constexpr):
   pid_batch = tl.program_id(0)
   pid_m = tl.program_id(1)

   offs_m = pid_batch * stride_x_batch + pid_m * stride_x_m
   block_N = tl.arange(0, BLOCK_N_SIZE)
   var = tl.zeros((BLOCK_N_SIZE,), tl.float32)

   # first loop over input tensor to compute the root mean of the square
   for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
       offs_n = block_n_start_idx + block_N
       x_ptr_mask = offs_n < N_SIZE
       # recompute address at each iteration
       x = tl.load(x_ptr + offs_m + offs_n * stride_x_k, mask=x_ptr_mask, other=0.0)
       var += tl.math.pow(x.to(tl.float32), 2)

   # we keep this reduction operation outside the loop for perf reasons
   var = tl.sum(var, axis=0) / N_SIZE
   rstd = tl.math.rsqrt(var + eps)

   # apply the normalization and multiply by RMS weights
   for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
       offs_n = block_n_start_idx + block_N
       x_ptr_mask = offs_n < N_SIZE
       rms_w = tl.load(rms_w_ptr + offs_n * stride_rms_w, mask=x_ptr_mask)

       x = tl.load(x_ptr + offs_m + offs_n * stride_x_k, mask=x_ptr_mask, other=0.0).to(tl.float32)
       x_hat = x * rstd
       out = x_hat * rms_w
       out_off = pid_batch * stride_out_batch + pid_m * stride_out_m + offs_n * stride_out_k
       tl.store(output_ptr + out_off, out, mask=x_ptr_mask)


def rmsnorm_triton_wrapper(x, rms_w, eps=1e-6):
    batch, M, K = x.shape
    assert rms_w.shape[-1] == K
    out = torch.empty_like(x)
    rmsnorm_triton[(batch, M,)](x, rms_w, out,
                                *x.stride(),
                                *rms_w.stride(),
                                *out.stride(),
                                N_SIZE=K, eps=eps, BLOCK_N_SIZE=1024,
                                )
    return out