import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np
import PIL
import shapely
import shapely.geometry
import torch as th
import torchvision as tv

from utils import dl_to_ld, get_files, ld_to_dl


def read_common_datafolder(path, annotation_filename):
    annotation_file_candidates = glob.glob(str(path / annotation_filename))
    assert len(annotation_file_candidates) == 1
    annotation_filename = annotation_file_candidates[0]
    annotation_filepath = path / annotation_filename
    with open(annotation_filepath) as annotation_file:
        annotation_data = json.load(annotation_file)

    n_inputs = len(annotation_data)
    annotation_data = ld_to_dl(annotation_data)
    # TODO: remove once fixed in dataset creation code
    if "filename" in annotation_data:
        annotation_data["image_filename"] = annotation_data.pop("filename")

    assert (
        len(set(get_files(path)).intersection(annotation_data["image_filename"]))
        == n_inputs
    )
    annotation_data = dl_to_ld(annotation_data)
    return annotation_data


def wkt_to_shapely(target, key="polygons"):
    target[key] = [shapely.geometry.shape(p) for p in target[key]]
    return target


def collate_fn(batch):
    # TODO: add support to pad images to minimum enclosing size
    # this gives support for different sized images and is the reason for the mask
    # in facebook-detr
    batch = list(zip(*batch))
    inputs = ld_to_dl(batch[0])
    for k in ["image", "mask", "segmentation"]:
        inputs[k] = th.stack(inputs[k])
    inputs["image_id"] = th.as_tensor(inputs["image_id"])
    targets = ld_to_dl(batch[1])
    batch[0] = inputs
    batch[1] = targets
    return batch


class CommonImageDataset(th.utils.data.Dataset):
    def __init__(
        self,
        path,
        target_filename: str = None,
        sample_data=None,
        label_map: dict = None,
        transformation=None,
        preremove_empty: bool = False,
        sample_preprocessor=None,
        image_loading_fn=None,
        add_segmentation: bool = False,
        image_postfix: str = "_RGB.jpg",
        segmentation_postfix: str = "_label.png",
    ):
        self.path = Path(path)
        self.label_map = label_map
        self.transformation = transformation
        self.preremove_empty = preremove_empty
        self.image_loading_fn = image_loading_fn if image_loading_fn else PIL.Image.open
        self.add_segmentation = add_segmentation

        assert target_filename is not None or sample_data is not None
        if target_filename is not None:
            self.sample_data = read_common_datafolder(path, target_filename)
            if self.label_map is None:
                label_map = dict.fromkeys(
                    np.concatenate(
                        [target["annotations"]["labels"] for target in self.sample_data]
                    )
                )
                self.label_map = {k: i for i, k in enumerate(label_map.keys())}

            def prepare(target):
                target["annotations"] = wkt_to_shapely(
                    target["annotations"], key="polygons"
                )
                # convert str label to numeric label
                target["annotations"]["labels"] = [
                    self.label_map[label] for label in target["annotations"]["labels"]
                ]
                return target

            self.sample_data = [prepare(target) for target in self.sample_data]
        elif sample_data is not None:
            self.sample_data = sample_data

        if add_segmentation:
            segmentation_files = glob.glob(str(path / ("*" + segmentation_postfix)))
            segmentation_files = [
                Path(fn).relative_to(path) for fn in segmentation_files
            ]

            self.segmentation_loading_fn = (
                image_loading_fn if "jpg" in segmentation_postfix else PIL.Image.open
            )
            for s in self.sample_data:
                image_fn = s["image_filename"]
                expected_segementation_fn = Path(
                    image_fn[: -len(image_postfix)] + segmentation_postfix
                )
                if expected_segementation_fn in segmentation_files:
                    s["segmentation_filename"] = expected_segementation_fn
                else:
                    raise Exception(
                        "no matching semgentation file", expected_segementation_fn
                    )

        if sample_preprocessor:
            self.sample_data = [
                sample_preprocessor(sample) for sample in self.sample_data
            ]

        self.n_empty_sample = sum(
            [len(sample["annotations"]["labels"]) == 0 for sample in self.sample_data]
        )
        self.n_nonempty_sample = len(self.sample_data) - self.n_empty_sample
        # TODO: remove this and replace by sample_preprocessor
        if preremove_empty and self.n_empty_sample > 0:
            self.sample_data = [
                target
                for target in self.sample_data
                if len(target["annotations"]["labels"]) > 0
            ]

    def get_sample_info(self):
        info = {
            "n_samples": len(self),
            "n_empty_samples": self.n_empty_sample,
        }
        return info

    def get_target_info(self):
        labels = []
        for sample in self.sample_data:
            labels += sample["annotations"]["labels"]
        counts = Counter(labels)
        counts = dict(sorted(counts.most_common()))
        counts = {f"n_targets_{k}": v for k, v in counts.items()}
        counts["n_targets"] = sum(counts.values())
        return counts

    def subset(self, indices):
        sample_data = [self.sample_data[i] for i in indices]
        clone = CommonImageDataset(
            self.path,
            sample_data=sample_data,
            label_map=self.label_map,
            transformation=self.transformation,
            preremove_empty=self.preremove_empty,
        )
        return clone

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
        if self.transformation:
            input, target = self.transformation(input, target)

        if "boxes" in target:
            if len(target["boxes"]):
                target["boxes"] = th.stack([th.as_tensor(g) for g in target["boxes"]])
            else:
                target["boxes"] = th.empty(0, 4)
        target["labels"] = th.as_tensor(target["labels"], dtype=int)
        target["difficults"] = th.as_tensor(target["difficults"], dtype=bool)
        target["image_id"] = index
        return input, target

    def __len__(self):
        return len(self.sample_data)
