"""
python process_data.py \
    --num_samples 500 \
    --test_samples 400 \
    --watermark_ratio 0.2
"""

import os
import json
import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Process original data and split prompts into training and test batches.")
    parser.add_argument("--input_file", type=str, default="data/alpaca_data.json", 
                        help="Path to original JSON file")
    parser.add_argument("--num_samples", type=int, default=100, 
                        help="Total number of prompts to sample for training split")
    parser.add_argument("--test_samples", type=int, default=50, 
                        help="Number of prompts to sample for final testing batch")
    parser.add_argument("--watermark_ratio", type=float, default=0.5, 
                        help="Fraction of training prompts to be watermarked (0.0 to 1.0)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for sampling")
    return parser.parse_args()

def load_prompts(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Extract the "instruction" field; if not present, fallback to "output".
    prompts = []
    for item in data:
        prompt = item.get("instruction", None)
        if prompt is None:
            prompt = item.get("output", "")
        prompts.append(prompt)
    return prompts

def write_jsonl(filename, prompts):
    with open(filename, 'w', encoding='utf-8') as f:
        for prompt in prompts:
            f.write(json.dumps({"prompt": prompt}) + "\n")

def main():
    args = parse_args()
    random.seed(args.seed)
    
    prompts = load_prompts(args.input_file)
    if len(prompts) < args.num_samples + args.test_samples:
        raise ValueError("Not enough prompts in the input file.")

    # Sample without replacement
    sampled = random.sample(prompts, args.num_samples + args.test_samples)
    training_samples = sampled[:args.num_samples]
    test_samples = sampled[args.num_samples:]
    
    # Split training samples into watermarked and non-watermarked based on ratio
    num_watermarked = int(args.num_samples * args.watermark_ratio)
    random.shuffle(training_samples)
    watermarked_prompts = training_samples[:num_watermarked]
    nonwatermarked_prompts = training_samples[num_watermarked:]
    
    # Ensure output directory exists
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_20")
    os.makedirs(out_dir, exist_ok=True)
    
    write_jsonl(os.path.join(out_dir, "train_watermarked.jsonl"), watermarked_prompts)
    write_jsonl(os.path.join(out_dir, "train_nonwatermarked.jsonl"), nonwatermarked_prompts)
    write_jsonl(os.path.join(out_dir, "final_test.jsonl"), test_samples)

    print(f"Training split: {len(watermarked_prompts)} watermarked, {len(nonwatermarked_prompts)} non-watermarked prompts.")
    print(f"Test split: {len(test_samples)} prompts.")
    
if __name__ == "__main__":
    main()
