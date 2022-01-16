# AttENT-Pytorch
Official Pytorch Implementation for AttENT.

## Updates
- *10/2021*: Our paper is accepted as the regular paper of BIBM 2021 and will give a presentation online (15 mins). The relevant code is released here.
- *08/2021*: Check out our paper [AttENT: Domain-Adaptive Medical Image Segmentation via Attention-Aware Translation and Adversarial Entropy Minimization] (submitted to BIBM 2021). With the aligned ensemble of attention-aware image pixel space and entropy-based feature space enables a well-trained segmentation modelto effectively transfer from source domain to target domain.

## Paper
![framework](https://github.com/lichen14/AttENT/blob/main/display/framework.png)

[Paper](https://github.com/lichen14/lichen14.github.io/assets/publication/BIBM_ATTENT.pdf) | [Slides](https://github.com/lichen14/AttENT/blob/main/display/BIBM_2021_Presentation-B221-AttENT.pdf)

If you find this code useful for your research, please cite our work:

```
@inproceedings{li2021attent,
  author={Li, Chen and Luo, Xin and Chen, Wei and He, Yulin and Wu, Mingfei and Tan, Yusong}, 
  booktitle={2021 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  title={AttENT: Domain-Adaptive Medical Image Segmentation via Attention-Aware Translation and Adversarial Entropy Minimization},
  year={2021},  
  volume={},  
  number={},  
  pages={952-959},  
  doi={10.1109/BIBM52615.2021.9669620}}
}
```
## Demo
![introduction](https://github.com/lichen14/AttENT/blob/main/display/introduction.png)

## Preparation
### Requirements

- Hardware: PC with NVIDIA 1080T GPU. (others are alternative.)
- Software: *Ubuntu 18.04*, *CUDA 10.0.130*, *pytorch 1.3.0*, *Python 3.6.9*
- Package:
  - `torchvision`
  - `tensorboardX`
  - `scikit-learn`
  - `glob`
  - `matplotlib`
  - `skimage`
  - `medpy`
  - `tqdm`
### Installation
0. Clone the repo:
```bash
$ git clone https://github.com/lichen14/AttENT
$ cd AttENT
```
1. Install OpenCV if you don't already have it:

```bash
$ conda install -c menpo opencv
```
2. Install this repository and the dependencies using pip:
```bash
$ pip install -e <root_dir>
```
With this, you can edit the TriDL code on the fly and import function 
and classes of AttENT in other project as well.
3. Optional. To uninstall this package, run:
```bash
$ pip uninstall AttENT
```

### Datasets
* By default, the datasets are put in ```<root_dir>/dataset```.
* An alternative option is to explicitlly specify the parameters ```DATA_DIRECTORY_SOURCE``` and ```DATA_DIRECTORY_TARGET``` in YML configuration files.
* After download, The MMWHS dataset directory should have this basic structure:
* **CHAOS**: Please follow the instructions [here](https://zenodo.org/record/3431873#.YSyWDC1JnfY/) to download images and semantic segmentation annotations.  We
extracted 20 labeled volumes from the T2-SPIR MRI training dataset.
* **MALBCV**: Please follow the instructions in [here](https://www.synapse.org/#!Synapse:syn3193805/) to download the images and ground-truths. We extracted 30 labeled volumes from the CT training dataset.
* The CHAOS and MALBCV datasets directory have the same structure but different content:
```bash
<root_dir>/data/CHAOS/mri_dataset/                               % MRI samples root
<root_dir>/data/CHAOS/mri_dataset/train/image/                   % MRI images
<root_dir>/data/CHAOS/mri_dataset/train/label/                   % MRI annotation
<root_dir>/data/CHAOS/mri_list/                                  % MRI samples list
...
<root_dir>/data/MALBCV/ct_dataset/                                % CT samples root
<root_dir>/data/MALBCV/ct_dataset/train/image/                    % CT images
<root_dir>/data/MALBCV/ct_dataset/train/label/                    % CT annotation
<root_dir>/data/MALBCV/ct_list/                                   % CT samples list
...
```
* For the common organs in the datasets, including liver, right kidney, left kidney, and spleen, we designed cross-modalities segmentation experiments between these organs.

### Pre-trained models
* Initial pre-trained model can be downloaded from [DeepLab-V2](https://drive.google.com/open?id=1TIrTmFKqEyf3pOKniv8-53m3v9SyBK0u). Please download it and put it [AttENT/pretrained_models/].
  
## Running the code
The well-trained model can be downloaded here [AttENT_deeplab](https://drive.google.com/open?id=1uNIydmPONNh29PeXqCb9MGRAnCWxAu99). You can use the pre-trained model or your own model to make a test as following:
```bash
$ cd <root_dir>/advent
$ python test.py --cfg ./configs/<your_yml_name>.yml --exp-suffix <your_define_suffix>
```
### Training adaptive segmenation network in AttENT
* To ensure reproduction, the random seed has been fixed in the code. Still, you may need to train a few times to reach the comparable performance.
* The whole training process is not in the end-end manner, which means there is two stages (1.Attention-aware Image Alignment and 2.Adversarial Feature Alignment) with human intervention when training the AttENT. And, there are two separate training directions (MRI \rightarrow CT and CT \rightarrow MRI).
* By default, logs and snapshots are stored in ```<root_dir>/experiments``` with this structure:
```bash
<root_dir>/experiments/logs
<root_dir>/experiments/snapshots  %output and trained model are stored in this file.
```

#### 1.Attention-aware Image Alignment in the Pixel Space
To perform cross-domain translation with Attention-aware CycleGAN:
```bash
$ python train.py --dataroot datasets/<dataset_name>/ --cuda
```
* We referred the implement of [PyTorch-CycleGAN](https://github.com/aitorzip/PyTorch-CycleGAN). Based on above [work](https://github.com/aitorzip/PyTorch-CycleGAN/blob/67da8f9e2b69bd68763451803c7700aaccc92f18/models.py#L3), we designed the Attention-aware CycleGAN, which is replaced by [Generator5](https://github.com/lichen14/AttENT/blob/main/attent/models.py). following the instruction of [PyTorch-CycleGAN](https://github.com/aitorzip/PyTorch-CycleGAN/blob/master/README.md), we revised the train.py with: 
```python
from models import Generator5
```
* With the Attention-aware CycleGAN, we obtained the translated multi-domain samples in the pixel space for the [2.Adversarial Feature Alignment in the Entropy Space].

#### 2.Adversarial Feature Alignment in the Entropy Space
To train AttENT separately from both directions: 
```bash
$ cd AttENT/attent
$ python train_MRI2CT.py --cfg ./configs/<your_yml_name>.yml  --exp-suffix <your_define_suffix>  --tensorboard         % using tensorboard
$ python train_CT2MRI.py --cfg ./configs/<your_yml_name>.yml  --exp-suffix <your_define_suffix>  --tensorboard         % using tensorboard
```


### Well-trained models and our implemented methods will be released soon.

## Acknowledgements
This codebase is heavily borrowed from [AdaptSegNet](https://github.com/wasidennis/AdaptSegNet), [PyTorch-CycleGAN](https://github.com/aitorzip/PyTorch-CycleGAN) and [ADVENT](https://github.com/valeoai/ADVENT).
Thanks to following repos for sharing and we referred some of their codes to construct AttENT:
### References
- [SIFA](https://github.com/cchen-cc/SIFA)
- [Attention-Gated-Networks](https://github.com/ozan-oktay/Attention-Gated-Networks)
- [SegmenTron](https://github.com/LikeLy-Journey/SegmenTron)
- [MICCAI-MMWHS-2017](http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/mmwhs/)
- [BDL](https://github.com/liyunsheng13/BDL)
