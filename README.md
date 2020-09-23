# Introduction
This repo is the a codebase of the Joint Detection and Embedding (JDE) model. JDE is a fast and high-performance multiple-object tracker that learns the object detection task and appearance embedding task simutaneously in a shared neural network. Due to the recent release of YOLOv5, we replace the detector in JDE with YOLOv5 and achieve high performance on MOT Benchmark. For some reasons, 
we can't release our stronger version, but we hope this repo will help researches/engineers to develop more practical real-time MOT systems.

# Requirements
Just follow the environmnet configuration of [YOLOv5](https://github.com/ultralytics/yolov5).

# Dataset zoo
Just follow the [DATASET_ZOO](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md) of JDE.


## Results on MOT16 Dataset
|  | MOTA | IDS |IDF1 | MOTP| FPS | Params(M) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| JDE(1088x608) |68.5 |1496 |66.8 | 0.221 |21 | 298 |
| Ours(1088x608) |71.0 |695 | 73.2 | 0.166 | 56 | 35 |

## Results on MOT20 Dataset
|  | MOTA | IDS |IDF1 | MOTP| FPS | Params(M) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| JDE(1088x608) |49.1 |24507 |38.4 | 0.272 |14 | 298 |
| Ours(1088x608) |55.3 |9190 | 47.5 | 0.287 | 24 | 35 |