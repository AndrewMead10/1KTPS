from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import time


def time_genration(length, model, input_ids, num_runs):
    start = time.time()
    for i in range(num_runs):
        output = model.generate(
            input_ids, max_length=length, min_length=length)
    end = time.time()
    return num_runs*length/(end - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=128,
                        help="Number of tokens to generate")
    parser.add_argument(
        "--model", type=str, default="EleutherAI/gpt-j-6B", help="Huggingface model to use")
    parser.add_argument("--dtype", type=str, default="fp16",
                        help="Data type to use (fp16, bf16, or int8)")
    parser.add_argument("--tf32", type=bool, default=True,
                        help="Use TF32 for matmul")
    parser.add_argument("--use_cache", type=bool,
                        default=True, help="Use cache for model")
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Number of runs to generate")

    args = parser.parse_args()

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    load_in_8bit = False
    if args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "int8":
        torch_dtype = torch.float16
        load_in_8bit = True

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, device_map='auto', use_cache=args.use_cache, load_in_8bit=load_in_8bit)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    text = "Hi how are you doing today?"

    input_ids = tokenizer(text, return_tensors="pt",
                          pad_to_multiple_of=64).input_ids

    # warmp up model
    _ = model.generate(input_ids, max_length=args.length,
                       min_length=args.length)

    print(time_genration(args.length, model, input_ids, args.num_runs))
