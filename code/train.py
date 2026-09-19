import argparse
import torch

from Utils import *
from Load_Dataset import *
from DISARM_class import *


def main(opts):
    print('\n--- load dataset ---')
    dataset = data_multi_aug(opts)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=opts.nThreads,
    )

    print('\n--- set saver ---')
    saver = Saver(opts)

    print('\n--- load model ---')
    model = DISARM(opts)
    model.setgpu(opts.gpu)
    if opts.resume is None:
        model.initialize()
        ep0 = -1
        total_it = 0
    else:
        ep0, total_it = model.resume(opts.resume)
    model.set_scheduler(opts, last_ep=ep0)
    ep0 += 1
    print('start the training at epoch %d' % ep0)

    print('\n--- train ---')
    stop_training = False

    for ep in range(ep0, opts.n_ep):
        for it, (images, c_org) in enumerate(train_loader):
            if images.size(0) != opts.batch_size:
                continue

            images = images.to(model.device).detach()
            c_org = c_org.to(model.device).detach()

            if opts.isDcontent:
                if (it + 1) % opts.d_iter != 0 and it < len(train_loader) - 2:
                    model.update_D_content(images, c_org)
                    continue
                model.update_D(images, c_org)
                model.update_EG()
            else:
                model.update_D(images, c_org)
                model.update_EG()

            if (total_it + 1) % opts.img_save_freq == 0:
                saver.write_img(-1, model)
            if (total_it + 1) % opts.model_save_freq == 0:
                saver.write_model(-1, total_it, model)

            print(
                'total_it: %d (ep %d, it %d), lr %08f'
                % (total_it, ep, it, model.gen_opt.param_groups[0]['lr'])
            )
            total_it += 1

            if total_it >= opts.max_iter:
                saver.write_img(-1, model)
                saver.write_model(-1, total_it, model)
                stop_training = True
                break

        if stop_training:
            break

    if opts.n_ep_decay > -1:
        model.update_lr()

    saver.write_img(ep, model)
    saver.write_model(ep, total_it, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training script')

    # data
    parser.add_argument('--dataroot', type=str, required=True, help='path to training data')
    parser.add_argument('--num_domains', type=int, default=5)
    parser.add_argument('--phase', type=str, default='train', help='data folder prefix')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--input_dim', type=int, default=1)
    parser.add_argument('--nThreads', type=int, default=8, help='number of data-loader workers')

    # output
    parser.add_argument('--name', type=str, default='trial', help='name of the output folder')
    parser.add_argument('--display_dir', type=str, default='./logs')
    parser.add_argument('--result_dir', type=str, default='./results')
    parser.add_argument('--display_freq', type=int, default=10)
    parser.add_argument('--img_save_freq', type=int, default=500, help='image preview frequency in training iterations')
    parser.add_argument('--model_save_freq', type=int, default=1000, help='checkpoint frequency in training iterations')

    # model and optimization
    parser.add_argument('--dis_scale', type=int, default=3)
    parser.add_argument('--dis_norm', type=str, default='None')
    parser.add_argument('--dis_spectral_norm', action='store_true')
    parser.add_argument('--nz', type=int, default=16, help='scanner latent dimension')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_dcontent', type=float, default=4e-5)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    parser.add_argument('--lr_policy', type=str, default='lambda')
    parser.add_argument('--n_ep', type=int, default=1200)
    parser.add_argument('--n_ep_decay', type=int, default=600)
    parser.add_argument('--max_iter', type=int, default=200000, help='maximum number of training iterations')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--d_iter', type=int, default=2, help='content-discriminator update interval')

    # loss weights
    parser.add_argument('--lambda_rec', type=float, default=10.0)
    parser.add_argument('--lambda_cc', type=float, default=10.0)
    parser.add_argument('--lambda_sf', type=float, default=7.0)
    parser.add_argument('--lambda_lat', type=float, default=8.0)
    parser.add_argument('--lambda_KL', type=float, default=0.01)
    parser.add_argument('--lambda_adv_b', type=float, default=1.0)
    parser.add_argument('--lambda_adv_s', type=float, default=1.0)
    parser.add_argument('--lambda_cls_D', '--lambda_cls', dest='lambda_cls_D', type=float, default=3.0)
    parser.add_argument('--lambda_cls_G', type=float, default=10.0)
    parser.add_argument('--lambda_content_l2', type=float, default=0.01)

    parser.add_argument('--isDcontent', action='store_true')
    parser.add_argument('--gpu', type=int, default=0, help='CUDA device index')

    main(parser.parse_args())
