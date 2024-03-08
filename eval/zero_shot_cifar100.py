#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""Computes zero-shot CIFAR100 accuracy using prompts from https://arxiv.org/abs/2103.00020.

Example usage:
    python zero_shot_cifar100.py --model-dir mofi-b16-hf --device cuda
"""
import argparse

import torch
import torchvision

from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from utils import get_mofi_transform


CIFAR100_PROMPTS = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a black and white photo of a {}.",
    "a low contrast photo of a {}.",
    "a high contrast photo of a {}.",
    "a bad photo of a {}.",
    "a good photo of a {}.",
    "a photo of a small {}.",
    "a photo of a big {}.",
    "a photo of the {}.",
    "a blurry photo of the {}.",
    "a black and white photo of the {}.",
    "a low contrast photo of the {}.",
    "a high contrast photo of the {}.",
    "a bad photo of the {}.",
    "a good photo of the {}.",
    "a photo of the small {}.",
    "a photo of the big {}.",
]

CIFAR_LABELS = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed",
    "bee", "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus",
    "butterfly", "camel", "can", "castle", "caterpillar", "cattle", "chair",
    "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile",
    "cup", "dinosaur", "dolphin", "elephant", "flatfish", "forest", "fox", "girl",
    "hamster", "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard",
    "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree",
    "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal",
    "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank",
    "telephone", "television", "tiger", "tractor", "train", "trout", "tulip",
    "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
]


def get_cifar100_class_embedding(model, batch_size, device):
    texts = []
    for text in CIFAR_LABELS:
        texts.extend([template.format(text.replace('_', ' ')) for template in CIFAR100_PROMPTS])

    embedding = []
    for i in tqdm(range(0, len(texts), batch_size)):
        input_ids = tokenizer(
            texts[i: i + batch_size],
            return_tensors='pt', padding=True, return_attention_mask=False,
        )['input_ids']
        with torch.no_grad():
            emb = model.get_text_features(input_ids=input_ids.to(device))
            emb /= emb.norm(dim=-1, keepdim=True)
        embedding.append(emb.detach())
    embedding = torch.concatenate(embedding)
    embedding = embedding.reshape(len(CIFAR_LABELS), len(CIFAR100_PROMPTS), -1).mean(-2)
    embedding /= embedding.norm(dim=-1, keepdim=True)
    return embedding


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, help='Path to MOFI model dir')
    parser.add_argument('--image-batch-size', type=int, default=16)
    parser.add_argument('--text-batch-size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    model = AutoModel.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model.eval()
    model.to(args.device)

    class_embedding = get_cifar100_class_embedding(model, args.text_batch_size, args.device)

    dataset = torchvision.datasets.CIFAR100('cifar100', download=True, train=False, transform=get_mofi_transform())
    ds = torch.utils.data.DataLoader(dataset, batch_size=args.image_batch_size, shuffle=False)

    predictions = []
    labels = []
    for imgs, label in tqdm(ds):
        with torch.no_grad():
            emb = model.get_image_features(imgs.to(args.device))
            predictions.append(emb.matmul(class_embedding.T).argmax(-1).detach().cpu())
            labels.append(label)

    predictions = torch.concatenate(predictions)
    labels = torch.concatenate(labels)
    accuracy = (predictions == labels).float().mean()

    print('accuracy@1:', accuracy.numpy())
