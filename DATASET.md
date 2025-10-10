# Dataset Preparation

The CVB Dataset can be download from [CVB](https://data.csiro.au/collection/csiro%3A58916v1)

The CVB-i Dataset can be download from[CVB-i](https://pan.baidu.com/s/1ikt4gIGKi1jhKh9xckmQjw) and the code is mbav.

## AVA format



Download and use 

utils/dataset_trans_tools/extract_rgb_frames_ffmpeg.sh

to extract frames with the following structure:

```
ava
|_ frames
|  |_ [video name 0]
|  |  |_ [video name 0]_000001.jpg
|  |  |_ [video name 0]_000002.jpg
|  |  |_ ...
|  |_ [video name 1]
|     |_ [video name 1]_000001.jpg
|     |_ [video name 1]_000002.jpg
|     |_ ...

```



