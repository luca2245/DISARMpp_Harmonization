# DISARM++: Towards Scanner-free Harmonization

DISARM++ is a 3D MRI harmonization model that separates anatomical content from scanner-related information. It can map T1-weighted MRI volumes either to a scanner-free space or to one of the scanner domains used during training.

The repository contains preprocessing, training and inference code, together with the configuration used for the experiments reported in the paper.

## Installation

```bash
git clone https://github.com/luca2245/DISARMpp_Harmonization.git
cd DISARMpp_Harmonization
pip install -r requirements.txt
```

Training and inference require PyTorch with a CUDA-enabled GPU. MRI preprocessing uses [FSL](https://fsl.fmrib.ox.ac.uk/fsl/docs/).

## Pretrained weights

The checkpoint used in the paper is available from the [pretrained-v1 release](https://github.com/luca2245/DISARMpp_Harmonization/releases/tag/pretrained-v1).

```bash
mkdir -p checkpoint
curl -L --fail \
  -o 'checkpoint/trained_disarm++.pth' \
  'https://github.com/luca2245/DISARMpp_Harmonization/releases/download/pretrained-v1/trained_disarm++.pth'
```

The same checkpoint is used for scanner-free inference and for transfer to the Gyroscan Intera reference domain.

## Preprocessing

Input images must be `.nii.gz` volumes. The preprocessing script applies:

1. `fslreorient2std`;
2. FAST bias-field correction;
3. 12-DOF FLIRT registration to MNI152.

The default options are the ones used in the paper: `MNI152_T1_1mm` and `normcorr`.

```bash
bash preprocessing/mri_prep.sh \
  /path/to/input \
  /path/to/intermediate \
  /path/to/preprocessed
```

A different reference resolution or FLIRT cost function can be selected explicitly, for example:

```bash
bash preprocessing/mri_prep.sh \
  /path/to/input \
  /path/to/intermediate \
  /path/to/preprocessed \
  --reference 2mm \
  --cost normmi
```

## Inference

### Scanner-free harmonization

```bash
python -u code/inference.py \
  --input_dir /path/to/preprocessed \
  --output_dir /path/to/output \
  --resume checkpoint/trained_disarm++.pth \
  --gpu 0 \
  --mode scanner-free \
  --pre_trained_model
```

With `--pre_trained_model`, the fixed 16-dimensional latent vector used in the paper is reused for every subject and every overlapping 3D window. The vector is reported in `PAPER_CONFIGURATION.md`.

Without `--pre_trained_model`, scanner-free inference samples a new Gaussian latent vector when the program starts.

### Sliding-window reconstruction

At inference, DISARM++ processes 3D windows of 26 consecutive slices with stride 1, resulting in a 25-slice overlap between consecutive windows. Each window is independently harmonized using the same scanner-free latent configuration.

The final volume is reconstructed by voxel-wise uniform averaging of all predictions covering the same spatial location. Thus, sufficiently interior slices receive predictions from up to 26 overlapping windows, while fewer predictions contribute near the volume boundaries.

[Sliding-window inference and reconstruction schematic (PDF)](figures/sliding_window_inference.pdf)

### Transfer to Gyroscan Intera

```bash
python -u code/inference.py \
  --input_dir /path/to/preprocessed \
  --output_dir /path/to/output \
  --resume checkpoint/trained_disarm++.pth \
  --gpu 0 \
  --mode reference \
  --pre_trained_model
```

The reference image is `example_ref_image/ref_gyroscan.nii.gz`. In the released checkpoint, Gyroscan Intera corresponds to domain index 3.

### Model trained on another dataset

For a separately trained model, omit `--pre_trained_model` and set `--num_domains` to the number of domains used during training. Reference-domain inference also requires the training-data root and the corresponding domain index:

```bash
python -u code/inference.py \
  --input_dir /path/to/preprocessed \
  --output_dir /path/to/output \
  --resume /path/to/model.pth \
  --gpu 0 \
  --num_domains 3 \
  --mode reference \
  --dataroot /path/to/training_data \
  --domain_idx 2
```

## Training

### Prepare the folders

The training loader expects NumPy volumes in folders named `trainA`, `trainB`, ..., one folder for each scanner domain. The supplied conversion script creates this layout from preprocessed NIfTI folders.

```bash
python code/prepare_training_data.py \
  --domain /path/to/domain_A \
  --domain /path/to/domain_B \
  --domain /path/to/domain_C \
  --domain /path/to/domain_D \
  --domain /path/to/domain_E \
  --output_dir /path/to/training_data
```

The order of the `--domain` arguments defines the domain labels. For the released model, the training domains were ordered as follows:

| Folder | Domain index | Scanner model |
|---|---:|---|
| `trainA` | 0 | Prisma Fit |
| `trainB` | 1 | Prisma |
| `trainC` | 2 | Achieva dStream |
| `trainD` | 3 | Gyroscan Intera |
| `trainE` | 4 | Intera |

`trainC` contains the Achieva dStream scans from both ADNI3 and PPMI. Each input folder should contain the preprocessed `.nii.gz` images for one scanner domain.

### Run training

The following settings reproduce the training configuration reported in the paper up to the main training update cycle used for the released checkpoint:

```bash
python -u code/train.py \
  --dataroot /path/to/training_data \
  --batch_size 2 \
  --num_domains 5 \
  --input_dim 1 \
  --result_dir model_savings \
  --display_dir model_logs \
  --d_iter 2 \
  --n_ep 1200 \
  --max_iter 69000 \
  --img_save_freq 500 \
  --model_save_freq 1000 \
  --isDcontent
```

The loss weights and optimizer defaults in `code/train.py` match the paper configuration. They are listed together in `PAPER_CONFIGURATION.md`. The `--max_iter` argument counts main training update cycles (`total_it`) and does not include the interleaved mini-batches used exclusively to update the anatomy discriminator. The learning rates remained fixed throughout the training run used to obtain the released checkpoint.

## Acknowledgments

This work builds on [DISARM](https://github.com/luca2245/DISARM_Harmonization) and [DRIT](https://github.com/HsinYingLee/DRIT).
