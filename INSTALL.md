# Installation
1.Clone this repo.
```shell
git clone https://github.com/Atten4Vis/LW-DETR.git
cd LW-DETR
```

2.Install Requirements
The code is developed and validated under ```python=3.8.20, pytorch=1.13.1, cuda=11.7```. Higher versions might be available as well.
- Python >= 3.8
- Numpy
- PyTorch >= 1.13
- torchvision>=0.14.1
- [fvcore](https://github.com/facebookresearch/fvcore/): `pip install 'git+https://github.com/facebookresearch/fvcore'`
- simplejson: `pip install simplejson`
- GCC >= 4.9
- PyAV: `conda install av -c conda-forge`
- ffmpeg (4.0 is prefereed, will be installed along with PyAV)
- PyYaml: (will be installed along with fvcore)
- tqdm: (will be installed along with fvcore)
- iopath: `pip install -U iopath` or `conda install -c iopath iopath`
- psutil: `pip install psutil`
- OpenCV: `pip install opencv-python`
- torchvision: `pip install torchvision` or `conda install torchvision -c pytorch`
- tensorboard: `pip install tensorboard`
- moviepy: (optional, for visualizing video on tensorboard) `conda install -c conda-forge moviepy` or `pip install moviepy`
- PyTorchVideo: `pip install pytorchvideo`
- [Detectron2](https://github.com/facebookresearch/detectron2):
- FairScale: `pip install 'git+https://github.com/facebookresearch/fairscale'`
- scipy: `pip install scipy`
- timm: `pip install timm`
- tqdm: `pip install tqdm`
- cython: `pip install cython`
- pycocotools: `pip install pycocotools`
```
    pip install -U torch torchvision cython
    pip install -U 'git+https://github.com/facebookresearch/fvcore.git' 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
    git clone https://github.com/facebookresearch/detectron2 detectron2_repo
    pip install -e detectron2_repo
    # You can find more details at https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md
```

3. Compiling CUDA operators
```shell
cd core/model/deform_attn
then run make.sh
# unit test (should see all checking is True)
python test.py
cd ../..
```


Add this repository to $PYTHONPATH.
```
export PYTHONPATH=/path/to/SlowFast/slowfast:$PYTHONPATH
```


