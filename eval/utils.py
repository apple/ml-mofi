#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import torch
from torchvision import transforms


def get_mofi_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224, antialias=False),
        transforms.CenterCrop(224),
        transforms.Normalize(
            mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]),
            std=torch.tensor([0.26862954, 0.26130258, 0.27577711]),
        ),
    ])
