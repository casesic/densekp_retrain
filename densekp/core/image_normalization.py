"""Tensor image denormalization used by current DenseKP logging."""

import torch


def denormalize_images(images):
    images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(
        1, 3, 1, 1
    )
    images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(
        1, 3, 1, 1
    )
    return images
