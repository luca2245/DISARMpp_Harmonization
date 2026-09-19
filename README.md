# DISARM++: Beyond Scanner-free Harmonization

MRI scans acquired across different sites and scanners can vary in intensity and contrast because of differences in acquisition protocols, hardware, and software. These variations can introduce unwanted scanner-related effects into multicenter analyses.

DISARM++ is a 3D MRI harmonization model that separates anatomical content from scanner-related information. It can map images either to a scanner-free space or to a reference scanner domain. This repository provides the code for inference and training, along with a pretrained checkpoint.

## Getting started

Clone the repository and install the Python dependencies:

```bash
git clone https://github.com/luca2245/DISARMpp_Harmonization.git
cd DISARMpp_Harmonization
pip install -r requirements.txt
```

Inference and training use PyTorch with a CUDA-enabled GPU. MRI preprocessing requires [FSL](https://fsl.fmrib.ox.ac.uk/fsl/docs/).

## Pretrained weights

The pretrained DISARM++ checkpoint is distributed through [GitHub Releases](https://github.com/luca2245/DISARMpp_Harmonization/releases/tag/pretrained-v1), rather than being stored in the repository.

Download [`trained_disarm++.pth`](https://github.com/luca2245/DISARMpp_Harmonization/releases/download/pretrained-v1/trained_disarm++.pth) and save it in a `checkpoint/` directory at the root of the repository. From the repository root, you can also download it with:

```bash
mkdir -p checkpoint
curl -L --fail \
  -o 'checkpoint/trained_disarm++.pth' \
  'https://github.com/luca2245/DISARMpp_Harmonization/releases/download/pretrained-v1/trained_disarm++.pth'
```

The same checkpoint is used for both pretrained harmonization modes: **scanner-free** and **reference** (Gyroscan Intera). The reference mode also uses the example image provided in `example_ref_image/ref_gyroscan.nii.gz`.

## MRI preprocessing

Input images must be in `.nii.gz` format. Preprocessing is required for both inference and training and consists of:

1. Reorientation with FSL `fslreorient2std`.
2. Bias-field correction with FSL `FAST`.
3. Registration to the MNI152 template with FSL `FLIRT`.

Run the preprocessing script from the repository root, providing separate input, intermediate, and output directories:

```bash
bash preprocessing/mri_prep.sh /path/to/input /path/to/intermediate /path/to/preprocessed
```

The script prompts you to choose the registration template and FLIRT cost function. For the configuration used here, select **4** (`MNI152_T1_1mm.nii.gz`) and **1** (`normcorr`), respectively. If registration fails, try another cost function available in the script.

## Inference

Run the following commands **from the repository root**. The pretrained reference mode loads `example_ref_image/ref_gyroscan.nii.gz` using a path relative to the current working directory.

### Pretrained model: scanner-free harmonization

```bash
python -u code/inference.py \
  --input_dir /path/to/preprocessed \
  --output_dir /path/to/output \
  --resume 'checkpoint/trained_disarm++.pth' \
  --gpu 0 \
  --mode scanner-free \
  --pre_trained_model
```

To harmonize to the Gyroscan Intera reference instead, change `--mode scanner-free` to `--mode reference`. Keep `--pre_trained_model` enabled; the pretrained reference mode uses the example reference image included in the repository.

The main inference arguments are:

- `--input_dir`: directory containing the preprocessed `.nii.gz` scans.
- `--output_dir`: directory for the harmonized images; output filenames are prefixed with `harm_`.
- `--resume`: path to the downloaded checkpoint.
- `--gpu`: CUDA device index (default: `0`).
- `--mode`: `scanner-free` or `reference`.
- `--pre_trained_model`: enables the pretrained inference configuration.

### Model trained on your own data

For a separately trained model, omit `--pre_trained_model`. For example, to harmonize to a reference domain from the training dataset:

```bash
python -u code/inference.py \
  --input_dir /path/to/preprocessed \
  --output_dir /path/to/output \
  --resume /path/to/trained_model.pth \
  --gpu 0 \
  --mode reference \
  --dataroot /path/to/data \
  --domain_idx 2
```

`--dataroot` should contain one subdirectory per scanner domain. In this mode, `--domain_idx` identifies the reference domain from which an image is selected. Neither argument is needed for scanner-free inference.

## Training

### Dataset setup

Organize the training data into scanner-domain subdirectories. The `create_scanner_folders()` function in `code/Load_Dataset.py` can be used to generate this structure; see the instructions in that file.

### Run training

From the repository root:

```bash
python -u code/train.py \
  --dataroot /path/to/data \
  --batch_size 2 \
  --num_domains 5 \
  --input_dim 1 \
  --result_dir model_savings \
  --display_dir model_logs \
  --d_iter 2 \
  --n_ep 240000 \
  --img_save_freq 500 \
  --model_save_freq 1000 \
  --isDcontent
```

`--num_domains` sets the number of scanner domains. Training outputs and checkpoints are written to `--result_dir`; `--img_save_freq` and `--model_save_freq` control how often image previews and checkpoints are saved, respectively.

## Acknowledgments

This work builds on [DISARM](https://github.com/luca2245/DISARM_Harmonization) and [DRIT](https://github.com/HsinYingLee/DRIT).
