"""
python main_watermark.py \
    --nwm_input processed_20/train_nonwatermarked.jsonl\
    --wm_input processed_20/train_watermarked.jsonl\
    --output_file training_data_20/merged_train.jsonl \
    --generator maryland
"""

import os
import json
import random
import torch
import argparse
from argparse import Namespace
from helpers.wm.generator import WmGenerator, OpenaiGenerator

from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM,
                          LogitsProcessorList)

def load_model(args):
    """Load and return the model and tokenizer"""

    args.is_seq2seq_model = any([(model_type in args.model_name_or_path) for model_type in ["t5","T0"]])
    args.is_decoder_only_model = any([(model_type in args.model_name_or_path) for model_type in ["gpt","opt","bloom"]])
    if args.is_seq2seq_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    elif args.is_decoder_only_model:
        if args.load_fp16:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,torch_dtype=torch.float16, device_map='auto')
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(f"Unknown model type: {args.model_name_or_path}")

    if args.use_gpu:
        device = "mps" if torch.mps.is_available() else "cpu"
        if args.load_fp16: 
            pass
        else: 
            model = model.to(device)
    else:
        device = "cpu"
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    return model, tokenizer, device

def parse_args():
    parser = argparse.ArgumentParser(description="Generate merged outputs preserving watermarked vs plain percentages")
    parser.add_argument("--wm_input", type=str, default="processed/train_watermarked.jsonl", 
                        help="Processed watermarked prompts file")
    parser.add_argument("--nwm_input", type=str, default="processed/train_nonwatermarked.jsonl", 
                        help="Processed non-watermarked prompts file")
    parser.add_argument("--output_file", type=str, default="training_data/merged_train.jsonl", 
                        help="Merged output file for finetuning")
    # New argument to select generator type.
    parser.add_argument("--generator", type=str, default="openai", choices=["openai", "stanford", "maryland"],
                        help="Select watermark generator: openai, stanford, or maryland")
    return parser.parse_args()

def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # Each line is a JSON with key "prompt"
    return [json.loads(line)["prompt"] for line in lines if line.strip()]

def main():
    print("Starting watermark generation...")  # Added progress update
    args = parse_args()
    # Load prompts from processed files.
    watermarked_prompts = load_jsonl(args.wm_input)
    nonwatermarked_prompts = load_jsonl(args.nwm_input)
    
    # Setup generation parameters.
    gen_args = Namespace(
        model_name_or_path="facebook/opt-125m",
        run_gradio=False,
        load_fp16=False,
        prompt_max_length=None,
        max_new_tokens=1000,
        generation_seed=123,
        use_sampling=True,
        n_beams=1,
        sampling_temp=0.7,
        use_gpu=True,
        seeding_scheme="hash",
        gamma=0.25,
        delta=2.0,
        normalizers="",
        ignore_repeated_bigrams=False,
        detection_z_threshold=4.0,
        select_green_tokens=True,
        skip_model_load=False,
        seed_separately=True
    )
    model, tokenizer, device = load_model(gen_args)
    
    # Instantiate the selected watermark generator.
    if args.generator == "openai":
        wm_generator = OpenaiGenerator(model, tokenizer, ngram=1, seed=gen_args.generation_seed, seeding=gen_args.seeding_scheme)
    elif args.generator == "stanford":
        from helpers.wm.generator import StanfordGenerator
        wm_generator = StanfordGenerator(model, tokenizer, ngram=1, seed=gen_args.generation_seed, seeding=gen_args.seeding_scheme)
    elif args.generator == "maryland":
        from helpers.wm.generator import MarylandGenerator
        wm_generator = MarylandGenerator(model, tokenizer, ngram=1, seed=gen_args.generation_seed, seeding=gen_args.seeding_scheme, gamma=gen_args.gamma, delta=gen_args.delta)
    
    # Instantiate plain non-watermarked generator.
    plain_generator = WmGenerator(model, tokenizer, ngram=1, seed=gen_args.generation_seed, seeding=gen_args.seeding_scheme)

    # New: global counter for processed prompts
    total_processed = 0
    checkpoint_dir = "model_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    merged_lines = []
    
    total_wm = len(watermarked_prompts)
    total_nwm = len(nonwatermarked_prompts)
    
    # Process watermarked prompts to generate watermarked outputs.
    for i, prompt in enumerate(watermarked_prompts, start=1):
        # Call watermark generator (expects a list of prompts)
        watermarked_out = wm_generator.generate([prompt],
                                max_gen_len=gen_args.max_new_tokens,
                                temperature=gen_args.sampling_temp,
                                top_p=0.95)[0]
        merged_lines.append(json.dumps({"prompt": prompt, "generated": watermarked_out}))
        print(f"Processed watermarked prompt {i}/{total_wm}")
        total_processed += 1
        if total_processed % 200 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{total_processed}")
            print(f"Saved model checkpoint at {checkpoint_path}")
            # New: save merged_lines checkpoint
            merged_checkpoint_path = os.path.join(checkpoint_dir, f"merged_checkpoint_{total_processed}.jsonl")
            with open(merged_checkpoint_path, "w", encoding="utf-8") as cp_f:
                cp_f.write("\n".join(merged_lines))
            print(f"Saved merged_lines checkpoint at {merged_checkpoint_path}")
    
    # Process non-watermarked prompts to generate plain outputs.
    for i, prompt in enumerate(nonwatermarked_prompts, start=1):
        plain_out = plain_generator.generate([prompt],
                                max_gen_len=gen_args.max_new_tokens,
                                temperature=gen_args.sampling_temp,
                                top_p=0.95)[0]
        merged_lines.append(json.dumps({"prompt": prompt, "generated": plain_out}))
        print(f"Processed plain prompt {i}/{total_nwm}")
        total_processed += 1
        if total_processed % 200 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{total_processed}")
            merged_checkpoint_path = os.path.join(checkpoint_dir, f"merged_checkpoint_{total_processed}.jsonl")
            with open(merged_checkpoint_path, "w", encoding="utf-8") as cp_f:
                cp_f.write("\n".join(merged_lines))
            print(f"Saved merged_lines checkpoint at {merged_checkpoint_path}")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Write merged output file.
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(merged_lines))
    
    print(f"Merged training file saved to {args.output_file}")

if __name__ == "__main__":
    main()
