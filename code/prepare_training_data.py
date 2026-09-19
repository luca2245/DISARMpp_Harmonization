import argparse
import os

import nibabel as nib
import numpy as np


def nii_basename(filename):
    if filename.endswith('.nii.gz'):
        return filename[:-7]
    return os.path.splitext(filename)[0]


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    for domain_idx, input_dir in enumerate(args.domain):
        domain_name = 'train' + chr(ord('A') + domain_idx)
        output_dir = os.path.join(args.output_dir, domain_name)
        os.makedirs(output_dir, exist_ok=True)

        files = sorted(
            f for f in os.listdir(input_dir)
            if f.endswith('.nii.gz')
        )
        if not files:
            raise ValueError(f'No .nii.gz files found in {input_dir}')

        for filename in files:
            source = os.path.join(input_dir, filename)
            image = nib.load(source).get_fdata(dtype=np.float32)
            if image.ndim != 3:
                raise ValueError(f'Expected a 3D volume, got shape {image.shape} for {source}')

            destination = os.path.join(output_dir, nii_basename(filename) + '.npy')
            if os.path.exists(destination) and not args.overwrite:
                raise FileExistsError(
                    f'{destination} already exists. Use --overwrite to replace existing files.'
                )
            np.save(destination, image)

        print(f'{domain_name}: {len(files)} volumes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert preprocessed NIfTI volumes to the trainA/trainB/... layout used by DISARM++.'
    )
    parser.add_argument(
        '--domain', action='append', required=True,
        help='folder containing preprocessed .nii.gz volumes; repeat once per domain in training order',
    )
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--overwrite', action='store_true')
    main(parser.parse_args())
