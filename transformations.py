import shapely
import torch as th
import torch.nn as nn
import torchvision as tv

from typing import List


def denormalize(
    tensor: th.Tensor, mean: List[float], std: List[float], inplace: bool = False
) -> th.Tensor:
    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = th.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = th.as_tensor(std, dtype=dtype, device=tensor.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor


class Indentity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return input, target


class TransformWrapper:
    def __init__(self, transform, apply_to: str = "input", key: str = "image"):
        self.transform = transform
        self.apply_to = apply_to
        self.key = key

    def __call__(self, input, target):
        if self.apply_to == "input":
            input[self.key] = self.transform(input[self.key])
        elif self.apply_to == "target":
            target[self.key] = self.transform(target[self.key])
        return input, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input, target):
        for tfm in self.transforms:
            input, target = tfm(input, target)
        return input, target
    

def resize(tensor, size, max_size=None):
    _, h, w = tensor.shape
    is_square = h == w
    if is_square:
        return tv.transforms.functional.resize(tensor, size)
    return tv.transforms.functional.resize(tensor, size-1, max_size=max_size)
    
class Resize(nn.Module):
    input_keys = ["image", "mask", "segmentation"]
    target_keys = ["polygons", "boxes"]

    def __init__(self, size, max_size=None):
        super().__init__()
        self.size = size
        self.max_size = max_size

    def __call__(self, inputs, targets):
        inputs = inputs.copy()
        _, origin_h, origin_w = inputs["image"].shape
        for key in self.input_keys:
            if key in inputs:
                inputs[key] = resize(inputs[key], self.size, self.max_size)
                
        targets = targets.copy()
        _, new_h, new_w = inputs["image"].shape
        scale_x, scale_y = new_w / origin_w, new_h / origin_h

        for key in self.target_keys:
            if key in targets:
                targets[key] = [
                    shapely.affinity.scale(g, scale_x, scale_y, origin=(0, 0))
                    for g in targets[key]
                ]
        return inputs, targets

# def pad(tensor, padded_size, fill_value=0, ignore_channels=False):
#     if tensor.shape[1:] != padded_size:
#         padded_tensor = th.full(
#             (tensor.shape[0], *padded_size),
#             fill_value,
#             dtype=tensor.dtype,
#             device=tensor.device,
#         )
#         padded_tensor[:, : tensor.shape[1], : tensor.shape[2]] = tensor
#         return padded_tensor
#     return tensor


# class Pad(nn.Module):
#     def __init__(self, padded_size, fill_value=0):
#         super().__init__()
#         self.padded_size = padded_size
#         self.fill_value = fill_value

#     def forward(self, input, target):
#         if input["image"].shape[1:] != self.padded_size:
#             input = input.copy()
#             for k in ["image", "mask", "segmentation"]:
#                 if k in input:
#                     input[k] = pad(input[k], self.padded_size, self.fill_value)
#         return input, target


class StandardNormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, input, target):
        input = input.copy()
        input["image"] = tv.transforms.functional.normalize(
            input["image"], self.mean, self.std
        )
        return input, target
