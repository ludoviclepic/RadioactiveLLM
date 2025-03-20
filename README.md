# Radioactive LLM Detection

The goal of this project is to check if a finetuned LLM is radioactive because it was trained on watermarked data from another LLM.

## Steps

1. **Download Data**
   ```bash
   mkdir data
   wget https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json -P data/
   ```

2. **Process Data**
   Run `process_data.py` to split the original JSON file into training and test batches.
   ```bash
   python process_data.py --num_samples 1000 --test_samples 500 --watermark_ratio 0.8
   ```

3. **Generate Watermarked Outputs**
   Run `main_watermark.py` to create generated answers preserving watermarked vs non-watermarked percentages.
   ```bash
   python main_watermark.py
   ```

4. **Finetune GPT2**
   Run `finetune_gpt2.py` to finetune a GPT2 model using both watermarked and non-watermarked training data.
   ```bash
   python finetune_gpt2.py
   ```

5. **Generate Outputs from Finetuned Model**
   Run `generate_from_finetune.py` to generate answers from the finetuned model.
   ```bash
   python generate_from_finetune.py --sample_size 0 --seed 42
   ```

6. **Check Radioactivity**
   This is the code based on our papper
   Run `main_reed_wm.py` to detect if the finetuned model's outputs are radioactive.
   ```bash
   python main_reed_wm.py \
    --model_name "finetuned_gpt2"\
    --dataset_path "training_data/generated_output_finegpt2.jsonl" \
    --method_detect maryland \
    --nsamples 500 \
    --batch_size 16 \
    --output_dir final_output_1/ \
    --ngram 4
   ```

## References

This project uses ideas and code from:
- [lm-watermarking](https://github.com/jwkirchenbauer/lm-watermarking)
- [three_bricks](https://github.com/facebookresearch/three_bricks/blob/main)
- [radioactive-watermark](https://github.com/facebookresearch/radioactive-watermark)
