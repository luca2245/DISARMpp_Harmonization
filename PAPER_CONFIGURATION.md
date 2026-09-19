# DISARM++ paper configuration

This file contains the configuration used for the experiments reported in the
DISARM++ paper.

## Model

- Training scanner domains: **5**
- Input channels: **1**
- Scanner latent dimension: **16**
- Subvolume depth: **26 slices**
- Batch size: **2**
- Training set: **702 T1-weighted MRI scans**

## Loss weights

| Term | Weight |
|---|---:|
| Reconstruction (`lambda_rec`) | 10 |
| Cross-cycle consistency (`lambda_cc`) | 10 |
| Scanner-free (`lambda_sf`) | 7 |
| Latent regression (`lambda_lat`) | 8 |
| KL divergence (`lambda_KL`) | 0.01 |
| Anatomical adversarial (`lambda_adv_b`) | 1 |
| Scanner adversarial (`lambda_adv_s`) | 1 |
| Scanner classification, discriminator (`lambda_cls_D`) | 3 |
| Scanner classification, generator (`lambda_cls_G`) | 10 |
| Anatomical latent L2 regularization | 0.01 |

## Optimization

Adam was used for all networks with:

- betas: **(0.5, 0.999)**
- weight decay: **1e-4**
- gradient clipping: **1.0**
- `d_iter = 2`

Learning rates:

| Component | Learning rate |
|---|---:|
| Anatomy encoder | 1e-4 |
| Scanner encoder | 1e-4 |
| Generator | 1e-4 |
| Scanner discriminator(s) | 1e-4 |
| Anatomy discriminator | 4e-5 |

The checkpoint used in the paper is the one saved at **69,000 training
iterations**.

## Scanner-free inference

The scanner-free latent can in general be sampled from `N(0, I)`. For the
experiments reported in the paper, one 16-dimensional realization was fixed
and used for all subjects:

```text
[ 0.3440, -0.5120,  0.6164,  0.0888,
 -0.2443,  1.6746,  0.1562, -1.2279,
 -0.3125,  0.4583,  1.5044, -1.2348,
  1.2667,  2.1365, -1.3497, -0.8333 ]
```

This is the vector used by `code/inference.py` with the
`--pre_trained_model` option. The null scanner-conditioning vector `c_0` is
used during scanner-free generation.

For a volume, inference is performed with **26-slice windows and stride 1**.
The same latent vector is used for every window, and overlapping predictions
are averaged to reconstruct the final 3D image. This is implemented in
`recompose_image_to_scannerfree()` in `code/Utils.py`.

## Preprocessing

The preprocessing used for the experiments is implemented in
`preprocessing/mri_prep.sh` and consists of:

1. reorientation with `fslreorient2std`;
2. bias-field correction with FSL FAST;
3. registration to `MNI152_T1_1mm` with FLIRT.

## Checkpoint

The pretrained model used in the paper is available at:

`checkpoint/trained_disarm++.pth`

Software dependencies and package versions are listed in `requirements.txt`.
Training and inference usage examples are provided in the main `README.md`.
