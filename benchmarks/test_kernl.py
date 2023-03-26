from kernl.model_optimization import optimize_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch._dynamo as torchdynamo
import time

torchdynamo.config.cache_size_limit = 512

tokenizer = AutoTokenizer.from_pretrained("AMead10/llama-7b", )

model = AutoModelForCausalLM.from_pretrained(
    "AMead10/llama-7b", torch_dtype=torch.float16, use_cache=True, load_in_8bit=False, use_auth_token=True)
model = model.eval().to('cuda:1')

text = "Hi how are you doing today?"

input_ids = tokenizer(text, return_tensors="pt",
                      pad_to_multiple_of=64).input_ids

input_ids = input_ids.to('cuda:1')


def warmup(model, input_ids, length):
    start = time.perf_counter()
    with torch.inference_mode():
        for i in range(10):
            model.generate(input_ids, max_length=length, min_length=length)
        torch.cuda.synchronize()

    return time.perf_counter() - start


def generate(model, input_ids, length):
    start = time.perf_counter()
    with torch.inference_mode(), torch.cuda.amp.autocast():
        output = model.generate(
            input_ids, max_length=length, min_length=length)
        torch.cuda.synchronize()
    end = time.perf_counter()

    return end - start


print(f'Warmup 128 : {warmup(model, input_ids, 128)}')
baseline128 = generate(model, input_ids, 128)
print(f'Baseline 128 : {baseline128/128}')
print(f'Warmup 512 : {warmup(model, input_ids, 512)}')
baseline512 = generate(model, input_ids, 512)
print(f'Baseline 512 : {baseline512/512}')

optimize_model(model)

print(f'Warmup 128 : {warmup(model, input_ids, 128)}')
optimized128 = generate(model, input_ids, 128)
print(f'Optimized 128 : {optimized128/128}')
print(f'Warmup 512 : {warmup(model, input_ids, 512)}')
optimized512 = generate(model, input_ids, 512)
print(f'Optimized 512 : {optimized512/512}')

print(f'Baseline 128 speedup: {baseline128/optimized128}')
print(f'Baseline 512 speedup: {baseline512/optimized512}')
