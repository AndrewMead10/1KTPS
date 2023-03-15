from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForCausalLM, pipeline
import torch

# WIP, stil needs to be tested, code should run


def sparse_gpt(W, p, B=64, Bs=2, lam=3):
    # W: layer matrix
    # p: sparsity percentage
    # B: lazy batch-update blocksize
    # Bs: adaptive mask selection blocksize
    # expected input shape: W: (drow, dcol)

    drow, dcol = W.shape

    # Initialize mask and error arrays
    M = torch.ones((drow, dcol), device='cuda')
    E = torch.zeros((drow, B), device='cuda')

    # Compute inverse Hessian
    H_inv = torch.inverse(
        2 * torch.matmul(W.transpose(0, 1), W) + lam * torch.eye(dcol, device='cuda'))

    H_inv_chol = torch.linalg.cholesky(H_inv, upper=True)

    # Prune weights in batches
    for i in range(0, dcol, B):
        for j in range(i, i+B):
            if j % Bs == 0:
                # Select mask of (1-p)% smallest weights
                weights = W[:, j:j+Bs]
                weights_norm_sq = torch.sum(weights ** 2, dim=0)
                errors = weights_norm_sq / (H_inv_chol[j, j] ** 2)
                _, idx = torch.topk(errors, int(Bs*(1-p)))
                mask = torch.zeros((drow, Bs))
                mask[:, idx] = 1
                M[:, j:j+Bs] = mask

            # Compute pruning errors
            e = W[:, j] / H_inv_chol[j, j]
            E[:, j-i] = e

            # Freeze weights that are not pruned
            E[:, j-i] = (1 - M[:, j]) * E[:, j-i]

            # Update weights in block
            E_block = E[:, j-i].unsqueeze(1)  # shape: (1024, 1)
            H_inv_block = H_inv_chol[j, j:i+B].unsqueeze(0)  # shape: (1, B)

            W[:, j:i+B] -= E_block * H_inv_block

        # Update remaining weights
        # print(f'W: {W[:, i+B:].shape}')
        # print(f'E: {E.shape}')
        # print(f'H: {H_inv_chol[i:i+B, i+B:].shape}')
        W[:, i+B:] -= (E @ H_inv_chol[i:i+B, i+B:])

    # Set pruned weights to zero
    W *= M
    return W


model = T5ForConditionalGeneration.from_pretrained(
    'google/flan-t5-small', device_map='auto')
weights = model.state_dict()

# Prune the weights using the SparseGPT algorithm
p = 0.5  # Set the sparsity percentage
B = 64  # Set the batch-update block size
Bs = 128  # Set the adaptive mask selection block size
for name, weight in weights.items():
    if 'weight' in name:
        if 'relative_attention_bias' in name or "layer_norm" in name or "embed_tokens" in name:
            continue
        # print(weight.shape)
        # print(name)
        weights[name] = sparse_gpt(weight, p)

# Set the pruned weights to the T5 model
model.load_state_dict(weights)
