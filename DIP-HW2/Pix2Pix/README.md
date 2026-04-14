## Implementation of Image-to-Image Translation with Fully Convolutional Network

This repository is Chang Huayijun's implementation of Assignment_02 of DIP.  
The project focuses on paired image-to-image translation on the Facades dataset using a Fully Convolutional Encoder-Decoder Network.

The current implementation follows the training framework of Pix2Pix-style paired translation, but the implemented model is a fully convolutional generator trained with pixel-wise reconstruction loss instead of a full adversarial GAN.


## Requirements

To install requirements:

```setup
python -m pip install -r requirements.txt
```

## Running

To download the Facades dataset and generate the training / validation file lists, run:

```basic
bash download_facades_dataset.sh
```

To train the model, run:

```point
python train.py
```

## Results

### Training and Validation Visualization
During training, the program periodically saves visualization results in:

```point
train_results/
val_results/
```

Each saved image contains three parts concatenated horizontally:
1. `Input image`
2. `Ground-truth target image`
3. `Model output image`

These results can be used to compare model performance on the training set and validation set.

### Model Checkpoints
The trained model weights are saved in:

```point
checkpoints/
```
The training script saves:

1. `A checkpoint every 50 epochs`

2. `The best model according to validation loss`


## Key Technical Details

### Network Structure
The current model is a fully convolutional encoder-decoder network implemented in `FCN_network.py`.

Encoder:
1. `Conv2d(3, 32, kernel_size=4, stride=2, padding=1) + BatchNorm2d(32) + ReLU`
2. `Conv2d(32, 64, kernel_size=4, stride=2, padding=1) + BatchNorm2d(64) + ReLU`
3. `Conv2d(64, 128, kernel_size=4, stride=2, padding=1) + BatchNorm2d(128) + ReLU`
4. `Conv2d(128, 256, kernel_size=4, stride=2, padding=1) + BatchNorm2d(256) + ReLU`
5. `Conv2d(256, 512, kernel_size=4, stride=2, padding=1) + BatchNorm2d(512) + ReLU`

Decoder:
1. `ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1) + BatchNorm2d(256) + ReLU`
2. `ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) + BatchNorm2d(128) + ReLU`
3. `ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1) + BatchNorm2d(64) + ReLU`
4. `ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1) + BatchNorm2d(64) + ReLU`
5. `ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1) + Tanh`


### Parameter Settings
The main training parameters in the current implementation are:

1. Optimizer: `Adam`
2. Learning rate: `0.0002`
3. Betas: `(0.5, 0.999)`
4. Loss function: `L1Loss`
5. Batch size: `16`
6. Number of epochs: `300`
7. Learning rate scheduler: `StepLR(step_size=200, gamma=0.2)`

The output layer uses `Tanh` because both input and target images are normalized to `[-1, 1]`.

### Dataset Split
The project uses the Facades dataset.

1. Training set: `train_list.txt`
2. Validation set: `val_list.txt`

According to the current dataset files:
- training images: `400`
- validation images: `100`

Each image is stored as a paired image. In `facades_dataset.py`, the image is split along the width dimension:
1. left half: input image
2. right half: target image

Each half has a resolution of `128 x 128`.

## Acknowledgement

>📋 Thanks for the algorithms proposed by 
- [Paper: Poisson Image Editing](https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf)
- [Paper: Image-to-Image Translation with Conditional Adversarial Nets](https://phillipi.github.io/pix2pix/)


