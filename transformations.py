import shapely
import torch as th
import torch.nn as nn
import torchvision as tv

# functional transformations ##################################################


def clip_polys(input, target):
    # clip polygons to image bounds
    if "polygons" in target:
        shape = input["image"].shape
        box = shapely.geometry.box(0, 0, shape[2], shape[1])
        target["polygons"] = [p.intersection(box) for p in target["polygons"]]
    return input, target


def polys_to_boxes(input, target):
    if "polygons" in target and not "boxes" in target:
        target["boxes"] = [
            p.minimum_rotated_rectangle.envelope for p in target["polygons"]
        ]
    return input, target


def drop_polys(input, target):
    # useful since polygons are not supported in current collate_fn
    if "polygons" in target:
        del target["polygons"]
    return input, target


def to_bounds(geometry):
    if geometry.is_empty or not geometry.is_valid:
        return (-1.0, -1.0, 0.0, 0.0)
    return geometry.bounds


def geometry_to_bounds(input, target, keys=["boxes", "polygons"]):
    for key in keys:
        if key in target:
            target[key] = [to_bounds(g) for g in target[key]]
    return input, target


# class transformations #######################################################


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


class Resize(tv.transforms.Resize):
    def forward(self, input, target):
        input = input.copy()
        target = target.copy()
        h, w = input["image"].shape[1:]
        scale_x, scale_y = self.size[1] / w, self.size[0] / h
        for k in ["image", "mask", "segmentation"]:
            if k in input:
                input[k] = super().forward(input[k])
        if "polygons" in target:
            target["polygons"] = [
                shapely.affinity.scale(g, scale_x, scale_y, origin=(0, 0))
                for g in target["polygons"]
            ]
        if "boxes" in target:
            target["boxes"] = [
                shapely.affinity.scale(g, scale_x, scale_y, origin=(0, 0))
                for g in target["boxes"]
            ]
        return input, target


def pad(tensor, padded_size, fill_value=0, ignore_channels=False):
    if tensor.shape[1:] != padded_size:
        padded_tensor = th.full(
            (tensor.shape[0], *padded_size),
            fill_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        padded_tensor[:, : tensor.shape[1], : tensor.shape[2]] = tensor
        return padded_tensor
    return tensor


class Pad(nn.Module):
    def __init__(self, padded_size, fill_value=0):
        super().__init__()
        self.padded_size = padded_size
        self.fill_value = fill_value

    def forward(self, input, target):
        if input["image"].shape[1:] != self.padded_size:
            input = input.copy()
            for k in ["image", "mask", "segmentation"]:
                if k in input:
                    input[k] = pad(input[k], self.padded_size, self.fill_value)
        return input, target


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


class RandomChoice(nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def forward(self, input, target):
        transform = self.transforms[int(th.randint(len(self.transforms), (1,)))]
        return transform(input, target)


class RandomTransform(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def transform(self, input, target):
        return input, target

    def forward(self, input, target):
        if th.rand(1) < self.p:
            input, target = self.transform(input, target)
        return input, target


class RandomHorizontalFlip(RandomTransform):
    def __init__(self, p: float = 0.5):
        super().__init__(p)

    def transform(self, input, target):
        input = input.copy()
        target = target.copy()
        for k in ["image", "mask", "segmentation"]:
            if k in input:
                input[k] = input[k].flip(2)
        if "polygons" in target:
            shape = input["image"].shape
            origin = (shape[2] / 2.0, shape[1] / 2.0)
            target["polygons"] = [
                shapely.affinity.scale(b, xfact=-1, origin=origin)
                for b in target["polygons"]
            ]
        return input, target


class RandomVerticalFlip(RandomTransform):
    def __init__(self, p: float = 0.5):
        super().__init__(p)

    def transform(self, input, target):
        input = input.copy()
        target = target.copy()
        for k in ["image", "mask", "segmentation"]:
            if k in input:
                input[k] = input[k].flip(1)
        if "polygons" in target:
            shape = input["image"].shape
            origin = (shape[2] / 2.0, shape[1] / 2.0)
            target["polygons"] = [
                shapely.affinity.scale(b, yfact=-1, origin=origin)
                for b in target["polygons"]
            ]
        return input, target


# class RandomCrop(tv.transforms.RandomCrop):
#     def forward(self, input, target):
#         assert "segmentation" not in input
#         input = input.copy()
#         target = target.copy()

# #         img = input["image"]
# #         if "mask" in input:
# #             msk = input["mask"]
#         tx, ty = 0, 0
#         if self.padding is not None:
#             for k in ["image", "mask", "segmentation"]:
#                 if k in input:
#                     input[k] = tv.transforms.functional.pad(input[k], self.padding, self.fill, self.padding_mode)
#             tx, ty = self.padding, self.padding

#         width, height = tv.transforms.functional._get_image_size(input["image"])
#         # pad the width if needed
#         if self.pad_if_needed and width < self.size[1]:
#             print("padw")
#             padding = [self.size[1] - width, 0]
#             tx = tx + padding[0]
#             img = tv.transforms.functional.pad(
#                 img, padding, self.fill, self.padding_mode
#             )
#         # pad the height if needed
#         if self.pad_if_needed and height < self.size[0]:
#             print("padh")
#             padding = [0, self.size[0] - height]
#             ty = ty + padding[1]
#             img = tv.transforms.functional.pad(
#                 img, padding, self.fill, self.padding_mode
#             )

#         i, j, h, w = self.get_params(img, self.size)
#         tx = tx - j
#         ty = ty - i
#         input["image"] = tv.transforms.functional.crop(img, i, j, h, w)
#         crop_box = shapely.geometry.box(-j, -i, w - 1, h - 1)
#         if "mask" in input:
#             input["mask"] = tv.transforms.functional.crop(msk, i, j, h, w)
#         if "polygons" in target:
#             target["polygons"] = [
#                 shapely.affinity.translate(g, tx, ty).intersection(crop_box)
#                 for g in target["polygons"]
#             ]
#         if "boxes" in target:
#             target["boxes"] = [
#                 shapely.affinity.translate(g, tx, ty) for g in target["boxes"]
#             ]
#         return input, target


class RandomScale(nn.Module):
    def __init__(
        self, scale=(0.08, 1.0), interpolation=tv.transforms.InterpolationMode.BILINEAR
    ):
        super().__init__()
        self.scale = scale
        self.interpolation = interpolation

    def forward(self, input, target):
        input = input.copy()
        target = target.copy()
        h, w = input["image"].shape[1:]
        scale = th.rand(1) * (self.scale[1] - self.scale[0]) + self.scale[0]
        h *= scale
        w *= scale

        for k in ["image", "mask", "segmentation"]:
            if k in input:
                input[k] = tv.transforms.functional.resize(
                    input[k], (int(h), int(w)), self.interpolation
                )
        if "polygons" in target:
            target["polygons"] = [
                shapely.affinity.scale(g, scale, scale, origin=(0, 0))
                for g in target["polygons"]
            ]
        if "boxes" in target:
            target["boxes"] = [
                shapely.affinity.scale(g, scale, scale, origin=(0, 0))
                for g in target["boxes"]
            ]
        return input, target


class ColorJitter(tv.transforms.ColorJitter):
    def forward(self, input, target):
        input = input.copy()
        input["image"] = super().forward(input["image"])
        return input, target
