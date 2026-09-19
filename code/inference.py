import argparse
import os

import nibabel as nib
import torch

from Utils import *
from Load_Dataset import *
from DISARM_class import *


PAPER_SCANNER_FREE_LATENT = torch.tensor([[
     0.3440, -0.5120,  0.6164,  0.0888,
    -0.2443,  1.6746,  0.1562, -1.2279,
    -0.3125,  0.4583,  1.5044, -1.2348,
     1.2667,  2.1365, -1.3497, -0.8333,
]], dtype=torch.float32)


def main(opts):
    os.makedirs(opts.output_dir, exist_ok=True)

    # Architecture used by the released checkpoint
    opts.num_domains = 5
    opts.input_dim = 1
    opts.dis_scale = 3
    opts.dis_norm = 'None'
    opts.dis_spectral_norm = False
    opts.phase = 'train'

    print('\n--- load model ---')
    model = DISARM(opts)
    model.setgpu(opts.gpu)
    model.resume(opts.resume, train=False)
    model.eval()

    img_list, filenames_list = load_nifti_images_from_folder(opts.input_dir)
    if not img_list:
        raise ValueError('No .nii.gz files found in the input directory.')

    if opts.mode == 'scanner-free':
        if opts.pre_trained_model:
            print('Harmonizing to scanner-free using the pretrained model...')
            z_rand = PAPER_SCANNER_FREE_LATENT.clone()
        else:
            print('Harmonizing to scanner-free...')
            z_rand = get_z_random(1, 16, random_type='gauss')

        har_data = transfer_img_list_to_scannerfree(img_list, z_rand, opts, model)

    else:
        if opts.pre_trained_model:
            print('Harmonizing to reference using the pretrained model...')
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ref_path = os.path.join(repo_root, 'example_ref_image', 'ref_gyroscan.nii.gz')
            ref_img = nib.load(ref_path).get_fdata()
            # Gyroscan Intera corresponds to domain index 3 in the released checkpoint.
            har_data = transfer_img_list_to_reference2(img_list, ref_img, 3, opts, model)
        else:
            if opts.dataroot is None:
                raise ValueError('--dataroot is required for reference inference with a custom model.')
            print('Harmonizing to reference...')
            har_data = transfer_img_list_to_reference(img_list, opts.domain_idx, opts, model)

    save_preprocessed_images(
        opts.output_dir,
        [nib_tensor2img(har_data[i]) for i in range(len(har_data))],
        filenames_list,
        source_dir=opts.input_dir,
        prefix='harm_',
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('--input_dir', type=str, required=True, help='folder containing preprocessed .nii.gz images')
    parser.add_argument('--dataroot', type=str, help='training dataset root, needed for custom reference inference')
    parser.add_argument('--output_dir', type=str, required=True, help='output folder')
    parser.add_argument('--resume', type=str, required=True, help='path to the trained checkpoint')
    parser.add_argument('--domain_idx', type=int, default=0, help='reference-domain index for a custom model')
    parser.add_argument('--gpu', type=int, default=0, help='CUDA device index')
    parser.add_argument(
        '--mode', type=str, default='scanner-free',
        choices=['scanner-free', 'reference'],
        help='harmonization mode',
    )
    parser.add_argument('--pre_trained_model', action='store_true', help='use the released checkpoint configuration')

    main(parser.parse_args())
