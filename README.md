# artifive-potsdam
tooling and demo for ArtifiVe-Potsdam dataset
http://rs.ipb.uni-bonn.de/data/artifive-potsdam/

Please check the notebook to see how the code can be used to create a pytorch dataset, to prepare the data with transformations, and some visual samples of the dataset content.

![potsdam image samples](potsdam_samples.png)

![artificial image samples](artificial_samples.png)

## Dataset


| dataset                          | training images | test images | training objects | test objects |
|----------------------------------|-----------------|-------------|------------------|--------------|
| fullsized                        | 24              | 14          | 6019             | 3833         |
| patched 600x600                  | 2400            | 1400        | 6978             | 4489         |
| patched 600x600 + 200 px overlap | 5400            | 3150        | 15379            | 9793         |
| artificial                       | 1000            |             | 10000            |              |

## Requirements

* numpy https://github.com/numpy/numpy
* pytorch https://github.com/pytorch/pytorch
* torchvision https://github.com/pytorch/vision
* pillow https://github.com/python-pillow/Pillow
* shapely https://github.com/Toblerity/Shapely
* matplotlib https://matplotlib.org/
* optional: libjpeg-turbo https://github.com/libjpeg-turbo/libjpeg-turbo/ + https://github.com/lilohuang/PyTurboJPEG

* TODO: add requirements file
 
## Benchmark details

### Training

* we train on patched/600x600_overlap200/training
* we split it into 70% for training and 30 % for validation, therefore the provided sample numbers in the paper are smaller than the actual number of images; also therfore your performance may vary
* we remove objects whose min and max sizes are outside of 20 and 200 px and require one side to be larger than 40 px
* we remove the empty images which further reduces the number of samples (`remove_empty` sample_filter)


### Testing

* we test on patched/600x600/test dataset
* we use pycocotools (https://github.com/cocodataset/cocoapi) to evaluate
* the baselines report the AP with IOU threshold 0.5