# artifive-potsdam
tooling and demo for ArtifiVe-Potsdam dataset
http://rs.ipb.uni-bonn.de/data/artifive-potsdam/

Please check the notebook to see how the code can be used to create a pytorch dataset, to prepare the data with transformations, and some visual samples of the dataset content.

# Benchmark details

Training

* we train on patched/600x600_overlap200/training
* we split it into 70% for training and 30 % for validation, therefore the provided sample numbers in the paper are smaller than the actual number of images
* we remove objects whose min and max sizes are outside of 20 and 200 px and require at least one side to be larger than 40 px
* we remove the empty images which further reduces the number of samples (`remove_empty` sample_filter)


Testing

* we test on patched/600x600/test dataset
* we use pycocotools (https://github.com/cocodataset/cocoapi) to evaluate
* the baselines report the AP with IOU threshold 0.5