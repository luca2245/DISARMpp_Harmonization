# DISARM++ paper configuration

This file collects the settings used for the experiments reported in the DISARM++ paper.

## Training data

The model was trained on 702 T1-weighted MRI scans from five scanner models:

- ADNI3: Prisma Fit (167), Prisma (69), Achieva dStream (26);
- PPMI: Achieva dStream (5);
- IXI: Gyroscan Intera (250), Intera (185).

The two Achieva dStream subsets are treated as the same scanner model, giving five training domains in total.

The domain order used to build the training set was:

| Folder | Domain index | Scanner model |
|---|---:|---|
| `trainA` | 0 | Prisma Fit |
| `trainB` | 1 | Prisma |
| `trainC` | 2 | Achieva dStream |
| `trainD` | 3 | Gyroscan Intera |
| `trainE` | 4 | Intera |

The `trainC` folder contains the Achieva dStream scans from both ADNI3 and PPMI.

## Model

- Input channels: **1**
- Scanner latent dimension: **16**
- Training subvolume depth: **26 slices**
- Batch size: **2**
- Scanner discriminator instances: **2**, with independent parameters

## Data transformation and augmentation

Training volumes are stored as 3D NumPy arrays. They are rescaled to `[-1, 1]` and a 26-slice subvolume is selected along the first spatial dimension used by the model.

The augmentation used during training is TorchIO `RandomElasticDeformation` with:

- `num_control_points = 10`
- `locked_borders = 2`
- `max_displacement = 8`
- probability **0.7**

The identity transform is selected with probability **0.3**.

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
| Anatomical latent L2 regularization (`lambda_content_l2`) | 0.01 |

These weights are exposed as command-line arguments in `code/train.py`; their defaults match the values above.

## Optimization

Adam is used for all networks with:

- `beta1 = 0.5`
- `beta2 = 0.999`
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

The checkpoint used for the paper was obtained after **69,000 training iterations**.

## Scanner-free inference

The scanner-free latent can in general be sampled from `N(0, I)`. In the paper experiments, one 16-dimensional realization was fixed and used for all subjects:

```text
[ 0.3440, -0.5120,  0.6164,  0.0888,
 -0.2443,  1.6746,  0.1562, -1.2279,
 -0.3125,  0.4583,  1.5044, -1.2348,
  1.2667,  2.1365, -1.3497, -0.8333 ]
```

This vector is defined in `code/inference.py` and is used when `--pre_trained_model` is enabled. The scanner condition is the null vector `c_0`.

Inference uses 26-slice windows with stride 1. The same latent vector is used for every window and overlapping predictions are averaged to reconstruct the 3D output. This is implemented in `recompose_image_to_scannerfree()` in `code/Utils.py`.

For reference-domain inference with the released checkpoint, Gyroscan Intera uses domain index **3** and the supplied `example_ref_image/ref_gyroscan.nii.gz` reference image.

## Preprocessing

The paper preprocessing is implemented in `preprocessing/mri_prep.sh` with the following defaults:

- `fslreorient2std`;
- FAST: `-t 1 -n 3 -H 0.1 -I 4 -l 20.0 --nopve -B -b`;
- FLIRT reference: `MNI152_T1_1mm.nii.gz`;
- FLIRT cost: `normcorr`;
- search range: `[-90, 90]` for x, y and z;
- degrees of freedom: **12**;
- interpolation: `trilinear`.

Package versions are listed in `requirements.txt`.
