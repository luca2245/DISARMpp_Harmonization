import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
from torch.optim import lr_scheduler
from torch.autograd import Variable
import matplotlib.colors as mcolors
import torch.nn as nn
from tqdm import tqdm
from Load_Dataset import *
from tensorboardX import SummaryWriter


def get_axis_slices(img, slice_index):
        num_rows = 1
        num_cols = 3
        names = ['Sagittal', 'Axial', 'Coronal']
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4))
     
        for j in range(num_cols):
                if(j == 0):
                    z_slice = img[:, :, slice_index]
                if(j == 1):
                    z_slice = img[slice_index, :, :]
                if(j == 2):
                    z_slice = img[:, (slice_index+18), :]
                # Plot the 2D slice in the current axis
                axes[j].imshow(z_slice, cmap='gray')
                axes[j].set_title(f'T1 - {names[j]}', fontsize=20)
        plt.tight_layout()
        plt.show()


def get_axis_slices_90_rot(img, slice_index):
        num_rows = 1
        num_cols = 3
        names = [ 'Sagittal','Coronal','Axial' ]
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4))
     
        for j in range(num_cols):
                if(j == 0):
                    z_slice = np.rot90(img[slice_index, :, :])
                if(j == 1):
                    z_slice = np.rot90(img[:, (slice_index+18), :])
                if(j == 2):
                    z_slice = np.rot90(img[:, :, slice_index])
                # Plot the 2D slice in the current axis
                axes[j].imshow(z_slice, cmap='gray')
                axes[j].set_title(f'T1 - {names[j]}', fontsize=20)
        plt.tight_layout()
        plt.show()

def tensor2img(img):
    img = img.cpu().float().detach().numpy()
    img = (img + 1) / 2.0 * 255.0
    return img.astype(np.uint8)

def nib_tensor2img(img):
    img = img.cpu().float().detach().numpy()
    img = (img + 1) / 2.0 * 255.0
    return img

def get_scheduler(optimizer, opts, cur_ep=-1):
  if opts.lr_policy == 'lambda':
    def lambda_rule(ep):
      lr_l = 1.0 - max(0, ep - opts.n_ep_decay) / float(opts.n_ep - opts.n_ep_decay + 1)
      return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=cur_ep)
  elif opts.lr_policy == 'step':
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opts.n_ep_decay, gamma=0.1, last_epoch=cur_ep)
  else:
    return NotImplementedError('no such learn rate policy')
  return scheduler


class Saver():
  def __init__(self, opts):
    self.display_dir = os.path.join(opts.display_dir, opts.name)
    self.model_dir = os.path.join(opts.result_dir, opts.name)
    self.image_dir = os.path.join(self.model_dir, 'images')
    self.display_freq = opts.display_freq
    self.img_save_freq = opts.img_save_freq
    self.model_save_freq = opts.model_save_freq

    # make directory
    if not os.path.exists(self.display_dir):
        os.makedirs(self.display_dir)
    if not os.path.exists(self.model_dir):
        os.makedirs(self.model_dir)
    if not os.path.exists(self.image_dir):
        os.makedirs(self.image_dir)

    # create tensorboard writer
    self.writer = SummaryWriter(log_dir=self.display_dir)

  # save result images
  def write_img(self, ep, model):
    if (ep + 1) % self.img_save_freq == 0:
        assembled_images1, assembled_images2, assembled_images3 = model.assemble_outputs()
        img_filename1 = '%s/gen_%05d_slice_1.jpg' % (self.image_dir, ep)
        img_filename2 = '%s/gen_%05d_slice_2.jpg' % (self.image_dir, ep)
        img_filename3 = '%s/gen_%05d_slice_3.jpg' % (self.image_dir, ep)
        pil_image1 = Image.fromarray(assembled_images1.squeeze(), mode='L')
        pil_image2 = Image.fromarray(assembled_images2.squeeze(), mode='L')
        pil_image3 = Image.fromarray(assembled_images3.squeeze(), mode='L')
        pil_image1.save(img_filename1)
        pil_image2.save(img_filename2)
        pil_image3.save(img_filename3)
    elif ep == -1:
        assembled_images1, assembled_images2, assembled_images3 = model.assemble_outputs()
        img_filename1 = '%s/gen_last_slice_1.jpg' % (self.image_dir, ep)
        img_filename2 = '%s/gen_last_slice_2.jpg' % (self.image_dir, ep)
        img_filename3 = '%s/gen_last_slice_3_.jpg' % (self.image_dir, ep)
        pil_image1 = Image.fromarray(assembled_images1.squeeze(), mode='L')
        pil_image2 = Image.fromarray(assembled_images2.squeeze(), mode='L')
        pil_image3 = Image.fromarray(assembled_images3.squeeze(), mode='L')
        pil_image1.save(img_filename1)
        pil_image2.save(img_filename2)
        pil_image3.save(img_filename3)

  # save model
  def write_model(self, ep, total_it, model):
    if (ep + 1) % self.model_save_freq == 1:
        print('--- save the model @ ep %d ---' % (ep))
        model.save('%s/%05d.pth' % (self.model_dir, ep), ep, total_it)
    elif ep == -1:
        model.save('%s/last.pth' % self.model_dir, ep, total_it)


# ### Utils for testing phase

def get_model_device(model):
    return next(model.parameters()).device


def get_z_random(batchSize, nz, random_type='gauss'):
    return torch.randn(batchSize, nz)


def transfer_to_scannerfree(source_img, opts, model):
    device = get_model_device(model)
    source_img_ = source_img.to(device=device, dtype=torch.float32)
    z_random = get_z_random(source_img.size(0), 16, 'gauss').to(device)
    with torch.no_grad():
        output_test = model.test_scannerfree_transfer(source_img_, z_random)
    return output_test[0, 0]


def recompose_image_to_reference(image, ref_scanner, opts, model, subset_size=26, moving_window=1):
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input image must be a torch tensor.")

    device = get_model_device(model)
    image = image.to(device=device, dtype=torch.float32)
    d, h, w = image.shape

    recomposed_image = torch.zeros((d, h, w), dtype=image.dtype, device=device)
    count = torch.zeros((d, h, w), dtype=torch.float32, device=device)

    dataset_dom = data_single_std(opts, ref_scanner)
    dataloader_dom = torch.utils.data.DataLoader(
        dataset_dom, batch_size=1, shuffle=True, num_workers=6
    )
    for batch in dataloader_dom:
        img, lab = batch

    img = img.to(device=device, dtype=torch.float32)
    lab = lab.to(device=device, dtype=torch.float32)

    start = 0
    while start + subset_size <= d:
        end = start + subset_size
        subset_image = image[start:end].unsqueeze(0).unsqueeze(0)
        img_ = img[:, :, start:end, :, :]

        with torch.no_grad():
            gen_subset_image = model.test_reference_transfer(
                image=subset_image, image_trg=img_, c_trg=lab
            )

        recomposed_image[start:end] += gen_subset_image[0, 0]
        count[start:end] += 1
        start += moving_window

    count = torch.where(count == 0, torch.ones_like(count), count)
    return recomposed_image / count


def base_recompose_image_to_reference(image, ref_img, ref_lab, opts, model, subset_size=26, moving_window=1):
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input image must be a torch tensor.")

    device = get_model_device(model)
    image = image.to(device=device, dtype=torch.float32)
    ref_img = ref_img.to(device=device, dtype=torch.float32)
    ref_lab = ref_lab.to(device=device, dtype=torch.float32)
    d, h, w = image.shape

    recomposed_image = torch.zeros((d, h, w), dtype=image.dtype, device=device)
    count = torch.zeros((d, h, w), dtype=torch.float32, device=device)

    start = 0
    while start + subset_size <= d:
        end = start + subset_size
        subset_image = image[start:end].unsqueeze(0).unsqueeze(0)
        img_ = ref_img[:, :, start:end, :, :]

        with torch.no_grad():
            gen_subset_image = model.test_reference_transfer(
                image=subset_image, image_trg=img_, c_trg=ref_lab
            )

        recomposed_image[start:end] += gen_subset_image[0, 0]
        count[start:end] += 1
        start += moving_window

    count = torch.where(count == 0, torch.ones_like(count), count)
    return recomposed_image / count


def custom_transform(x):
    return x.unsqueeze(0)


def custom_permute(x):
    return x.permute(0, 2, 3, 1)


def transfer_img_list_to_reference(img_list, ref_scanner, opts, model):
    list_new = []
    device = get_model_device(model)

    transforms = Compose([
        ToTensor(),
        custom_transform,
        custom_permute,
        tio.RescaleIntensity((-1, 1)),
    ])

    img_list = [transforms(img).unsqueeze(0) for img in img_list]
    dataset_dom = data_single_std(opts, ref_scanner)
    dataloader_dom = torch.utils.data.DataLoader(
        dataset_dom, batch_size=1, shuffle=True, num_workers=8
    )
    for batch in dataloader_dom:
        ref_img, ref_lab = batch

    for i in tqdm(range(len(img_list)), desc="Harmonizing images to reference scanner"):
        img_ = img_list[i][0, 0].to(device=device, dtype=torch.float32)
        output = base_recompose_image_to_reference(
            img_, ref_img, ref_lab, opts, model, subset_size=26, moving_window=1
        )
        list_new.append(output)
    return list_new


def transfer_img_list_to_reference2(img_list, ref_img, ref_scanner, opts, model):
    list_new = []
    device = get_model_device(model)

    transforms = Compose([
        ToTensor(),
        custom_transform,
        custom_permute,
        tio.RescaleIntensity((-1, 1)),
    ])

    img_list = [transforms(img).unsqueeze(0) for img in img_list]
    ref_img = transforms(ref_img).unsqueeze(0).to(device=device, dtype=torch.float32)
    ref_lab = torch.zeros((1, opts.num_domains), dtype=torch.float32, device=device)
    ref_lab[0, ref_scanner] = 1.0

    for i in tqdm(range(len(img_list)), desc="Harmonizing images to reference scanner"):
        img_ = img_list[i][0, 0].to(device=device, dtype=torch.float32)
        output = base_recompose_image_to_reference(
            img_, ref_img, ref_lab, opts, model, subset_size=26, moving_window=1
        )
        list_new.append(output)
    return list_new


def recompose_image_to_scannerfree(image, opts, model, z_rand, subset_size=26, moving_window=1):
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input image must be a torch tensor.")

    device = get_model_device(model)
    image = image.to(device=device, dtype=torch.float32)
    z_random = z_rand.to(device=device, dtype=torch.float32)
    d, h, w = image.shape

    recomposed_image = torch.zeros((d, h, w), dtype=image.dtype, device=device)
    count = torch.zeros((d, h, w), dtype=torch.float32, device=device)

    start = 0
    while start + subset_size <= d:
        end = start + subset_size
        subset_image = image[start:end].unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            gen_subset_image = model.test_scannerfree_transfer(subset_image, z_random)

        recomposed_image[start:end] += gen_subset_image[0, 0]
        count[start:end] += 1
        start += moving_window

    count = torch.where(count == 0, torch.ones_like(count), count)
    return recomposed_image / count


def transfer_img_list_to_scannerfree(img_list, z_rand, opts, model):
    list_new = []
    device = get_model_device(model)

    transforms = Compose([
        ToTensor(),
        custom_transform,
        custom_permute,
        tio.RescaleIntensity((-1, 1)),
    ])
    img_list = [transforms(img).unsqueeze(0) for img in img_list]

    for i in tqdm(range(len(img_list)), desc="Harmonizing images to scanner-free"):
        img_ = img_list[i][0, 0].to(device=device, dtype=torch.float32)
        output = recompose_image_to_scannerfree(
            img_, opts, model, z_rand, subset_size=26, moving_window=1
        )
        list_new.append(output)
    return list_new


def load_nifti_images_from_folder(folder_path):
    nifti_images = []
    filenames = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".nii.gz"):
            file_path = os.path.join(folder_path, filename)
            img = nib.load(file_path)
            nifti_images.append(img.get_fdata())
            filenames.append(filename)

    return nifti_images, filenames

# Save harmonized volumes in the same voxel space as their respective inputs.
def save_preprocessed_images(folder_path, nifti_images, filenames, source_dir, prefix="harm_"):
    if len(nifti_images) != len(filenames):
        raise ValueError("The number of output volumes and input filenames must match.")

    os.makedirs(folder_path, exist_ok=True)

    for img_array, filename in zip(nifti_images, filenames):
        source_img = nib.load(os.path.join(source_dir, filename))
        img_array = np.asarray(img_array, dtype=np.float32)

        if img_array.shape != source_img.shape:
            raise ValueError(
                f"Shape mismatch for {filename}: output {img_array.shape}, "
                f"input {source_img.shape}. Cannot reuse the input geometry."
            )

        # Copy spatial metadata, but store the harmonized intensities as float32
        # without applying the input image's original intensity scaling.
        header = source_img.header.copy()
        header.set_data_dtype(np.float32)
        header.set_slope_inter(1.0, 0.0)

        new_img = nib.Nifti1Image(img_array, affine=source_img.affine, header=header)
        new_file_path = os.path.join(folder_path, prefix + filename)
        nib.save(new_img, new_file_path)

        print(f"Saved preprocessed image: {new_file_path}")

