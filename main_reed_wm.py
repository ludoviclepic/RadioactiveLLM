# This code is adapted from the repository: https://github.com/facebookresearch/three_bricks
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
python main_reed_wm.py \
    --model_name "finetuned_gpt2"\
    --dataset_path "training_data/generated_output_finegpt2.jsonl" \
    --method_detect maryland \
    --nsamples 400 \
    --batch_size 16 \
    --output_dir final_output/ \
    --ngram 4
"""


import random
import argparse
import os
import time
import json

import tqdm
import pandas as pd
import numpy as np

import torch
from helpers.utils import HiddenPrints 
with HiddenPrints():
    from peft import PeftModel    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sentence_transformers import SentenceTransformer

from helpers.wm import (WmGenerator, OpenaiGenerator, MarylandGenerator, StanfordGenerator,
                WmDetector,  OpenaiDetector, MarylandDetector, StanfordDetector, 
                MarylandDetectorZ, OpenaiDetectorZ)

import helpers.utils


def get_args_parser():
    parser = argparse.ArgumentParser('Args', add_help=False)

    # model parameters
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--adapters_name', type=str)

    # prompts parameters
    parser.add_argument('--dataset_path', type=str, default="")
    parser.add_argument('--fake_seed', type=int, default=0)

    # generation parameters
    parser.add_argument('--seq_length', type=int, default=256)

    # watermark parameters
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seeding', type=str, default='hash', help='seeding method for rng key generation as introduced in https://github.com/jwkirchenbauer/lm-watermarking')
    parser.add_argument('--hash_key', type=int, default=35317, help='hash key for rng key generation')
    parser.add_argument('--method_detect', type=str, default='openai', help='Statistical test to detect watermark. Choose between: same (same as method), openai, openaiz, maryland, marylandz')
    parser.add_argument('--ngram', type=int, default=2, help='n-gram size for rng key generation')
    parser.add_argument('--topk', type=int, default=1, help='top-logits to look at')
    parser.add_argument('--gamma', type=float, default=0.25, help='gamma for maryland: proportion of greenlist tokens')
    parser.add_argument('--delta', type=float, default=2.0, help='delta for maryland: bias to add to greenlist tokens')
    parser.add_argument('--v1', type=int, default=0, help='wheter to use deduplication of watermarked windows + current token (v2) instead of watermarked window (v1). ')
    parser.add_argument('--no_dedup', type=int, default=0, help='SETTING TO 1 will make results false!')

    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature: lower values reduce noise by making the distribution sharper')

    # expe parameters
    parser.add_argument('--nsamples', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--do_eval', type=helpers.utils.bool_inst, default=True)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--split', type=int, default=None)
    parser.add_argument('--nsplits', type=int, default=None)

    # distributed parameters
    parser.add_argument('--ngpus', type=int, default=None)

    return parser


import json
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
class TextDataset(Dataset):
    def __init__(self, filename, tokenizer, seed, seq_length=256, indicies_lines=None):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.texts = []
        self.indicies_lines = indicies_lines
        if (self.indicies_lines is None):
            print("carreful: indicies_lines is None")
        with open(filename, 'r') as file:
            lines = [json.loads(line) for line in file]
            random.Random(seed).shuffle(lines)
        for i, line in enumerate(lines):
            if (i>=self.indicies_lines[0] and i<self.indicies_lines[1]):
                data = line
                # Updated key selection to work with jsonl having "generated" field
                if "generated" in data:
                    text = data["generated"]
                elif "output" in data:
                    text = data["output"]
                elif "result" in data:
                    text = data["result"]
                else:
                    text = data["text"]
                tokens = tokenizer.tokenize(text)
                self.texts.extend(tokens)
                
    
    def __len__(self):
        return len(self.texts) // self.seq_length
    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = (idx + 1) * self.seq_length
        tokens = self.texts[start:end]
        return self.tokenizer.convert_tokens_to_ids(tokens)

# The TextDataset class reads the input JSONL file line-by-line.
# Each line (a JSON object) is loaded and shuffled using a given seed.
# For each record within the specified range, it selects the text by prioritizing the "generated" field 
# (falling back to "output", "result", or "text" if needed), tokenizes the text,
# and appends the tokens to a list.
# The __len__ method computes the number of fixed-length token sequences available,
# and __getitem__ returns the token IDs for a slice of tokens of length seq_length.


def main(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Define device: use MPS if available, otherwise CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # build model
    model_name = args.model_name
    # Automatically add tokenizer files if missing
    vocab_file = os.path.join(model_name, "vocab.json")
    merges_file = os.path.join(model_name, "merges.txt")
    if not (os.path.exists(vocab_file) and os.path.exists(merges_file)):
        print("Tokenizer files not found. Saving GPT2 tokenizer to the model folder...")
        tmp_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tmp_tokenizer.save_pretrained(model_name)
    # Now load your tokenizer from the model directory
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    adapters_name = args.adapters_name
    args.ngpus = torch.cuda.device_count() if args.ngpus is None else args.ngpus
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        max_memory={i: '32000MB' for i in range(args.ngpus)}, # automatically handles the number of gpus
        offload_folder="offload",
        local_files_only=False,  # force download if model files not found locally
        trust_remote_code=True   # allow execution of code from the model repository
    ).to(device)
    if adapters_name is not None:
        print(f"Loading adapter {adapters_name}")
        model = PeftModel.from_pretrained(model, adapters_name).to(device)
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print(f"Using device: {device}")

    # load wm detector
    assert (args.method_detect in ["openai", "maryland"])
    if args.method_detect == "openai":
        detector = OpenaiDetector(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key)
    elif args.method_detect == "maryland":
        detector = MarylandDetector(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, gamma=args.gamma, delta=args.delta)


    # do splits
    with open(args.dataset_path, "r") as f:
        nprompts = sum(1 for l in f)
    if args.split is not None:
        left = nprompts * args.split // args.nsplits 
        right = nprompts * (args.split + 1) // args.nsplits if (args.split != args.nsplits - 1) else nprompts
        print(f"Creating prompts from {left} to {right}")
    else:
        (left, right) = (0, nprompts)
        print(f"Creating prompts from {left} to {right} (all prompts)")
    
    # (re)start experiment
    os.makedirs(args.output_dir, exist_ok=True)
    start_point = 0 # if resuming, start from the last line of the file
    if os.path.exists(os.path.join(args.output_dir, f"results_kgrams.jsonl")):
        with open(os.path.join(args.output_dir, f"results_kgrams.jsonl"), "r") as f:
            for _ in f:
                start_point += 1
    print(f"Starting from {start_point}")
    import zlib
    dataset = TextDataset(args.dataset_path, tokenizer, args.fake_seed, seq_length=args.seq_length, indicies_lines=(left, right))

    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    # generate
    all_times = []
    
    from torch.nn import CrossEntropyLoss
    tokenizer.pad_token_id = 0
    loss_fct = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction="none")
    num_iter = args.nsamples//args.batch_size
    dic = {}
    dic_r = {}
    # check if file does not already exist
    if not os.path.exists(os.path.join(args.output_dir, f"results_kgrams.jsonl")):
        with open(os.path.join(args.output_dir, f"results_kgrams.jsonl"), "a") as f:
            for i, x in enumerate(dataloader):
                if i %10 == 0:
                    print(f"iteraton: {i}")
                if i >num_iter:
                    break
                # generate chunk
                current_batch = {bs:set() for bs in range(args.batch_size)}
                with torch.inference_mode():
                    x = torch.stack(x, 1).to(device)
                    decoded_sentences = tokenizer.batch_decode(x, skip_special_tokens=True)
                    fwd = model(x)
                    logits = fwd[0][:,:-1,:]
                    # Scale logits by temperature to control noise: lower temperature means less randomness.
                    logits = logits / args.temperature
                    target = x[:, 1:]
                    # outputs = torch.argmax(logits ,-1)
                    values, outputs = torch.topk(logits,args.topk, dim=-1)
                    # Define the value of k
                    k = args.ngram
                    # Loop over the range from 0 to 255-k
                    for j in range(target.size()[-1]-k): # deduplication is done at the batch level
                        # Slice a from i to i+k and b at i+k, then concatenate along the last dimension. creater watermark window + current token made of watermark window from the input and current token predicted by the suspected model
                        slices = [torch.cat((target[:, j:j+k], outputs[:, j+k, top].unsqueeze(-1)), dim=-1) for top in range(args.topk)] 
                        slices = [torch.stack([slices[a][b] for a in range(args.topk)]).tolist() for b in range(len(slices[0]))]
                        # slices = [slice.tolist() for slice in slices]
                        for num_slice, slice in enumerate(slices):
                            l = len(slice)
                            for t, tuple in enumerate(slice):
                                # This is where dedup happens: we check that the watermark window is not part of the batch
                                if not str(tuple[:-1]) in current_batch[num_slice] or args.no_dedup: # if deduplication is off, we add all the tuples
                                    tup = tuple if args.v1==0 else tuple[:-1] # We also either add the watermark window + current token to the set of seen tuples (or just the watermark window)
                                    if t==l-1:
                                        current_batch[num_slice].add(str(tuple[:-1]))
                                    if not str(tup) in dic.keys():
                                        seed = detector.get_seed_rng(tuple[:-1])
                                        detector.rng.manual_seed(seed)
                                        if args.method_detect == "openai":
                                            rs = torch.rand(detector.vocab_size, generator=detector.rng) # n
                                            rt = -(1 - rs).log()[tuple[-1]]
                                            r = rt.item()#detector.score_tok(tuple[:-1], tuple[-1])
                                        elif args.method_detect == "maryland":
                                            vocab_permutation = torch.randperm(detector.vocab_size, generator=detector.rng)
                                            greenlist = vocab_permutation[:int(detector.gamma * detector.vocab_size)] # gamma * n are in greenlist
                                            rt = 1 if tuple[-1] in greenlist else 0
                                            r = rt
                                        dic[str(tup)] = r
                                        dic_r[str(tup)] = rs[tuple[-1]].item() if args.method_detect == "openai" else r 
                    mean_r = np.mean(list(dic_r.values()))
                    nb_disinct = len(dic.values())
                    pvalues = detector.get_pvalues_by_t(dic.values())
                    mean_ = np.mean([pvalue if pvalue > 0 else 0 for pvalue in pvalues])
                    log_pvalues = [np.log10(pvalue) if pvalue > 0 else -0.43 for pvalue in pvalues]
                # log
                f.write(json.dumps({
                    "iteration": i, 
                    'p_value': pvalues[-1] if len(pvalues)>0 else 0,
                    "mean_r":mean_r,
                    'log10_pvalue': log_pvalues[-1],
                    "nb_disinct":nb_disinct,
                    "nb":i*args.batch_size*args.seq_length
                    }) + "\n")
                f.flush()




if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
