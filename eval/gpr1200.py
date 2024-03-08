#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""Generates embedding which can be used for GPR1200 evaluation.

Example usage:

  python gpr1200.py --model-dir mofi-b16-hf --dataset-path images --device cuda --output mofi_b16

Eval output using `evaluate.py` from "https://github.com/Visual-Computing/GPR1200":

  python evaluate.py  --evalfile-path mofi_b16.npy --dataset-path images
"""

import argparse
import os
import torch

from tqdm import tqdm
from transformers import AutoModel
import numpy as np
from PIL import Image

from utils import get_mofi_transform


class GPR1200Dataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, transform):
        names = sorted(os.listdir(images_dir), key=lambda name: int(name.split("_")[0]))
        self.files = [os.path.join(images_dir, name) for name in names]
        self.transform = transform
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert('RGB')
        return self.transform(img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, help='Path to MOFI model dir')
    parser.add_argument('--dataset-path', type=str, help='Path to GPR1200 images dir')
    parser.add_argument('--output', type=str, help='Output path for the computed embedding')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    dataset = GPR1200Dataset(args.dataset_path, get_mofi_transform())
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = AutoModel.from_pretrained(args.model_dir)
    model.eval()
    model.to(args.device)

    embedding = []
    for imgs in tqdm(loader):
        with torch.no_grad():
            emb = model.get_image_features(imgs.to(args.device))
            emb /= emb.norm(dim=-1, keepdim=True)
        embedding.append(emb.detach().cpu().numpy())
    embedding = np.concatenate(embedding, 0)
    np.save(args.output, embedding)
