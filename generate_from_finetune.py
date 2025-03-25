"""
python generate_from_finetune.py\
    --input_file processed_20/final_test.jsonl\
    --output_file training_data_20/generated_output_finegpt2.jsonl\
    --sample_size 400\
    --seed 42
"""

import json
import os
import argparse
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from tqdm import tqdm  # added import for progress bar

def parse_args():
    parser = argparse.ArgumentParser(description="Generate answers from the fine-tuned GPT2 model using prompts from a JSONL file")
    parser.add_argument("--input_file", type=str, default="processed/final_test.jsonl", required=True, 
                        help="Input JSONL file with prompts")
    parser.add_argument("--output_file", type=str, default="training_data/generated_finetuned_gpt2.jsonl", 
                        help="Output JSONL file with generated answers")
    parser.add_argument("--sample_size", type=int, default=0, 
                        help="Number of samples to randomly select from the input file (0 means use all lines)")
    parser.add_argument("--seed", type=int, default=None, 
                        help="Random seed for reproducible sampling")
    return parser.parse_args()

def main():
    args = parse_args()
    model = GPT2LMHeadModel.from_pretrained("./finetuned_gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("./finetuned_gpt2")
    model.eval()
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model.to(device)

    output_lines = []
    with open(args.input_file, "r", encoding="utf-8") as f_in:
        first_char = f_in.read(1)
        f_in.seek(0)
        if first_char == "[":
            data_list = json.load(f_in)
            all_lines = [json.dumps(item) for item in data_list]
        else:
            all_lines = f_in.readlines()
    # Apply sampling if sample_size > 0 and seed provided
    if args.seed is not None:
        random.seed(args.seed)
    if args.sample_size > 0 and args.sample_size < len(all_lines):
        all_lines = random.sample(all_lines, args.sample_size)
    
    for line in tqdm(all_lines, desc="Generating outputs"):  # updated loop with progress bar
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Skipping invalid JSON line: {line.strip()}")
            continue
        prompt = data["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            inputs["input_ids"], 
            max_new_tokens=200,  # ensure generated tokens do not exceed model limits
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True, 
            temperature=0.2
        )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_lines.append(json.dumps({"prompt": prompt, "generated": generated}))
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    with open(args.output_file, "w", encoding="utf-8") as f_out:
        f_out.write("\n".join(output_lines))
    
    print(f"Generated outputs saved to {args.output_file}")

if __name__ == "__main__":
    main()
