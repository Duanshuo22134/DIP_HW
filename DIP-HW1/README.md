## Implementation of Image Geometric Transformation

This repository is Chang Huayijun's implementation of Assignment_01 of DIP - Image Warping and Point-Based Deformation.


## Requirements

To install requirements:

```setup
python -m pip install -r requirements.txt
```

## Running

To run basic transformation, run:

```basic
python run_global_transform.py
```

To run point guided transformation, run:

```point
python run_point_transform.py
```

## Results

### Basic Transformation - Image Geometric Transformation
<img src="pics/global_transform.gif" alt="global transformation demo" width="800">

The demo shows composition of scale, rotation, translation transformations performed around image center.

### Point Guided Deformation - MLS Affine Deformation
<img src="pics/point_transform.gif" alt="point-guided deformation demo" width="800">

The demo shows intuitive point-guided image warping using Moving Least Squares algorithm.


## Key Technical Details

### MLS Algorithm Steps:
1. Compute weights: $w_i = 1/(\|v - p_i\|^{2\alpha} + \varepsilon)$
2. Calculate weighted centroids: $p^*, q^*$
3. Center control points: $\hat{p}_i = p_i - p^*, \hat{q}_i = q_i - q^*$
4. Compute covariance matrices using einsum
5. Solve linear system: $M = W^{-1}J$
6. Apply transformation with reverse mapping
7. Use bilinear interpolation for sampling

## Acknowledgement

>📋 Thanks for the algorithms proposed by [Image Deformation Using Moving Least Squares](https://people.engr.tamu.edu/schaefer/research/mls.pdf).

**Authors of Original Paper**: Scott Schaefer, Travis McPhail, Joe Warren (Rice University)
