## CycleGAN

"Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"

Paper: https://arxiv.org/pdf/1703.10593.pdf

### Abstract
Image-to-image translation is a class of vision and graphics problems where the goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs. However, for many tasks, paired training data will not be available. We present an approach for learning to translate an image from a source domain X to a target domain Y in the absence of paired examples. Our goal is to learn a mapping G:X→Y such that the distribution of images from G(X) is indistinguishable from the distribution Y using an adversarial loss. Because this mapping is highly under-constrained, we couple it with an inverse mapping F:Y→X and introduce a cycle consistency loss to push F(G(X))≈X (and vice versa). Qualitative results are presented on several tasks where paired training data does not exist, including collection style transfer, object transfiguration, season transfer, photo enhancement, etc. Quantitative comparisons against several prior methods demonstrate the superiority of our approach.

## Workflow
### Convert Domain A to B (Horse to Zebra)
![Domain A to B](https://github.com/Shumway82/CycleGAN/blob/master/CycleGAN/images/model.jpg)
#
### Convert Domain B to A (Zebra to Horse)
![Domain B to A](https://github.com/Shumway82/CycleGAN/blob/master/CycleGAN/images/model1.jpg)


## Modifications
* Modify the ResNet-Generator with recursive residual blocks like in the DRRN. http://cvlab.cse.msu.edu/pdfs/Tai_Yang_Liu_CVPR2017.pdf
* Option to use an U-Net with skip connections.
* An an perceptual loss witch is extracted and calculated from the discriminator network of the corresponding domain A or B. 

## Results
* Coming soon

## Pre-requiremtents
* tensorflow >= 1.8 
* pil 
* numpy 

## Installation tf_base package
1. Clone the repository
```
$ git clone git@github.com:Shumway82/tf_core.git
```
2. Go to folder
```
$ cd tf_core
```
3. Install with pip3
```
$ pip3 install tfcore
or for editing the repository 
$ pip3 install -e .
```

## Install CycleGAN package

1. Clone the repository
```
$ git@github.com:Shumway82/CycleGAN.git
```
2. Go to folder
```
$ cd CycleGAN
```
3. Install with pip3
```
$ pip3 install -e .
```

## Usage-Example

1. Training
```
$ python pipeline_trainer.py --dataset "../horse2zebra/" --config_path "../config/" 
```

2. Inferencing
```
$ python pipeline_inferencer.py --dataset "../horse2zebra/testA" --outdir ../Results/Horse2Zebra_AtoB" --model_dir ../pretrained_models/generator/Horse2Zebra/"
```
