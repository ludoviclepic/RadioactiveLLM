import argparse
import subprocess
import sys

def run_command(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Command failed: {cmd}")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="Run the full pipeline sequentially.")
    # Arguments to pass extra flags for each script
    parser.add_argument("--process_data_args", type=str, 
                        default="--num_samples 500 \
                                 --test_samples 400 \
                                 --watermark_ratio 0.8", 
                        help="Extra args for process_data.py")
    parser.add_argument("--main_watermark_args", type=str, 
                        default="--nwm_input processed/train_nonwatermarked.jsonl \
                                 --output_file training_data/merged_train.jsonl \
                                 --generator maryland", 
                        help="Extra args for main_watermark.py")
    parser.add_argument("--finetune_gpt2_args", type=str, 
                        default="", 
                        help="Extra args for finetune_gpt2.py")
    parser.add_argument("--generate_from_finetune_args", type=str, 
                        default="--input_file processed/final_test.jsonl \
                                 --output_file training_data/generated_output_finegpt2.jsonl \
                                 --sample_size 400 \
                                 --seed 42", 
                        help="Extra args for generate_from_finetune.py")
    parser.add_argument("--main_reed_wm_args", type=str, 
                        default="--model_name finetuned_gpt2 \
                                 --dataset_path training_data/generated_output_finegpt2.jsonl \
                                 --method_detect maryland \
                                 --nsamples 400 \
                                 --batch_size 1 \
                                 --output_dir final_output_1/ \
                                 --ngram 4", 
                        help="Extra args for main_reed_wm.py")
    args = parser.parse_args()

    # Step 1: process_data.py
    run_command(f"python process_data.py {args.process_data_args}")

    # Step 2: main_watermark.py
    run_command(f"python main_watermark.py {args.main_watermark_args}")

    # Step 3: finetune_gpt2.py
    run_command(f"python finetune_gpt2.py {args.finetune_gpt2_args}")

    # Step 4: generate_from_finetune.py
    run_command(f"python generate_from_finetune.py {args.generate_from_finetune_args}")

    # Step 5: main_reed_wm.py
    run_command(f"python main_reed_wm.py {args.main_reed_wm_args}")

if __name__ == "__main__":
    main()
