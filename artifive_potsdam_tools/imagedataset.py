import glob
import json
import os
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import PIL
import shapely
import shapely.geometry
import torch as th
import torchvision as tv


def ld_to_dl(lst):
    # list of dicts to dict of lists
    return {key: [dic[key] for dic in lst] for key in lst[0]}


def read_datafolder(path, sample_file):
    sample_file_candidates = list(path.glob(sample_file))
    assert len(sample_file_candidates) == 1
    sample_file = sample_file_candidates[0]
    sample_file = path / sample_file
    with open(sample_file) as annotation_file:
        sample_data = json.load(annotation_file)

    n_inputs = len(sample_data)
    # check if sample data filenames are contained in path
    sample_data_dl = ld_to_dl(sample_data)
    path_files = os.listdir(path)
    found_images = set(path_files).intersection(sample_data_dl["image_filename"])
    assert len(found_images) == n_inputs
    if "segmentation_filename" in sample_data_dl:
        found_segmentations = set(path_files).intersection(
            sample_data_dl["segmentation_filename"]
        )
        assert len(found_segmentations) == n_inputs
    return sample_data


def stack_n_pad_tensors(tensors):
    max_shape = th.as_tensor([list(tensor.shape) for tensor in tensors]).max(0)[0]
    stack_shape = [len(tensors)] + max_shape.tolist()
    stack = th.zeros(stack_shape, dtype=tensors[0].dtype, device=tensors[0].device)
    for tn, sl in zip(tensors, stack):
        sl[: tn.shape[0], : tn.shape[1], : tn.shape[2]].copy_(tn)
    return stack


def collate_fn(
    batch,
    stackable_inputs=["image", "mask", "segmentation"],
    stackable_targets=[],
    tensorable_inputs=["image_id"],
    tensorable_targets=[],
):
    batch = list(zip(*batch))
    inputs = ld_to_dl(batch[0])
    for k in stackable_inputs:
        if k in inputs:
            inputs[k] = stack_n_pad_tensors(inputs[k])
    for k in tensorable_inputs:
        if k in inputs:
            inputs[k] = th.as_tensor(inputs[k])
    targets = ld_to_dl(batch[1])
    for k in stackable_targets:
        if k in targets:
            targets[k] = stack_n_pad_tensors(targets[k])
    for k in tensorable_targets:
        if k in targets:
            targets[k] = th.as_tensor(targets[k])
    return [inputs, targets]


def to_bounds(geometry):
    if geometry.is_empty or not geometry.is_valid:
        return (0.0, 0.0, 0.0, 0.0)
    return geometry.bounds


def count_empty_samples(samples):
    def __is_empty(sample):
        if sample["annotations"]:
            return len(sample["annotations"]["labels"]) == 0
        return True

    return sum([__is_empty(sample) for sample in samples])


class ImageDataset(th.utils.data.Dataset):
    def __init__(
        self,
        path,
        sample_filename: str = None,
        sample_data=None,
        sample_file_filter=None,
        label_map: dict = None,
        transform=None,
        sample_preprocessor=None,
        sample_filter=None,
        image_loading_fn=None,
        add_segmentation: bool = False,
    ):
        self.path = Path(path)
        self.label_map = label_map
        self.transform = transform
        self.image_loading_fn = image_loading_fn if image_loading_fn else PIL.Image.open
        self.add_segmentation = add_segmentation

        if sample_filename is not None:
            self.sample_data = read_datafolder(path, sample_filename)

            def __prep_polys(sample):
                polygons = sample["annotations"]["polygons"]
                polygons = [shapely.geometry.shape(p) for p in polygons]
                sample["annotations"]["polygons"] = polygons
                return sample

            self.sample_data = [__prep_polys(smp) for smp in self.sample_data]
        elif sample_data is not None:
            self.sample_data = sample_data
        elif sample_file_filter is not None:
            assert isinstance(sample_file_filter, str)
            filenames = glob.glob(str(path / sample_file_filter))
            self.sample_data = [
                {"image_filename": fn, "annotations": {}} for fn in filenames
            ]
        else:
            raise NotImplementedError("not a single required sample input provided.")

        if add_segmentation:
            sample_filename = self.sample_data[0]["segmentation_filename"]
            segmentation_format = sample_filename.split(".")[-1]
            if "jpg" in segmentation_format:
                self.segmentation_loading_fn = image_loading_fn
            else:
                self.segmentation_loading_fn = PIL.Image.open

        if sample_preprocessor:
            self.sample_data = [sample_preprocessor(smp) for smp in self.sample_data]
        if sample_filter:
            self.sample_data = [smp for smp in self.sample_data if sample_filter(smp)]
        if self.label_map is None:
            # build label map
            labels = [smp["annotations"]["labels"] for smp in self.sample_data]
            unique_labels = set(np.concatenate(labels))
            self.label_map = {k: i for i, k in enumerate(unique_labels)}

    def get_sample(self, idx):
        return self.sample_data[idx]

    def get_sample_info(self):
        n_empty_sample = count_empty_samples(self.sample_data)
        info = {
            "n_samples": len(self),
            "n_empty_samples": n_empty_sample,
        }
        return info

    def get_target_info(self):
        labels = []
        for sample in self.sample_data:
            labels += sample["annotations"]["labels"]
        counts = dict(Counter(labels))
        counts = OrderedDict(sorted(counts.items()))
        counts["n_targets"] = sum(counts.values())
        return counts

    def subset(self, indices):
        print("ImageDatset.subset() is depreceated")
        sample_data = [self.sample_data[i] for i in indices]
        clone = ImageDataset(
            self.path,
            sample_data=sample_data,
            label_map=self.label_map,
            transform=self.transform,
            add_segmentation=self.add_segmentation,
        )
        return clone

    def prep_boxes(self, boxes):
        if len(boxes):
            boxes = th.stack([th.as_tensor(g) for g in boxes])
        else:
            boxes = th.empty(0, 4)
        return boxes

    def prep_labels(self, labels):
        numeric_labels = [self.label_map[lbl] for lbl in labels]
        return th.as_tensor(numeric_labels, dtype=int)

    def __getitem__(self, index):
        input = {}
        image_filename = self.sample_data[index]["image_filename"]
        image_filepath = self.path / image_filename
        image = self.image_loading_fn(image_filepath)
        image = tv.transforms.functional.to_tensor(image)
        input["image"] = image
        input["mask"] = th.ones(1, *image.shape[1:])
        input["image_size"] = image.shape
        input["image_id"] = index
        input["image_filename"] = image_filename
        if self.add_segmentation:
            segmentation_filename = self.sample_data[index]["segmentation_filename"]
            segmentation_filepath = self.path / segmentation_filename
            segmentation = self.segmentation_loading_fn(segmentation_filepath)
            segmentation = tv.transforms.functional.to_tensor(segmentation)
            input["segmentation"] = segmentation

        target = self.sample_data[index]["annotations"].copy()
        if self.transform:
            input, target = self.transform(input, target)

        if "boxes" in target:
            target["boxes"] = self.prep_boxes(target["boxes"])
        if "labels" in target:
            target["labels"] = self.prep_labels(target["labels"])
        if "is_difficult" in target:
            target["is_difficult"] = th.as_tensor(target["is_difficult"], dtype=bool)
        target["image_id"] = index
        return input, target

    def __len__(self):
        return len(self.sample_data)
