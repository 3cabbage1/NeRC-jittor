import os
import time
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import jittor as jt
from jittor import init
from jittor import nn
from shutil import copyfile
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.fields import Pts_Bias, RenderingNetwork, RenderingNetworkM, SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer
import math

class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False):
        # self.device = 'cuda'

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'])
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.nerf_outside = NeRF(**self.conf['model.nerf'])
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network'])
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network'])
        self.color_network = RenderingNetworkM(**self.conf['model.rendering_network'])
        # self.pts_bias = Pts_Bias().to(self.device)
        # self.color_network2 = RenderingNetworkM(**self.conf['model.rendering_network']).to(self.device)
        
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())
        # params_to_train += list(self.pts_bias.parameters())
        # params_to_train += list(self.color_network2.parameters())

        self.optimizer = jt.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     None,
                                     self.color_network,
                                     None,
                                     **self.conf['model.neus_renderer'])

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

    def train(self):
        # self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()

        for iter_i in tqdm(range(res_step)):
            # print(image_perm[self.iter_step % len(image_perm)])
            data = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], self.batch_size)

            rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
            rays_ld = data[:, 10:]

            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

            background_rgb = None
            if self.use_white_bkgd:
                background_rgb = jt.ones([1, 3])

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = jt.ones_like(mask)

            mask_sum = mask.sum() + 1e-5
            render_out = self.renderer.render(rays_o, rays_d, near, far,
                                              background_rgb=background_rgb,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),rays_ld = rays_ld)

            color_fine = render_out['colorm_fine']
            colorm_fine = render_out['colorm_fine']

            s_val = render_out['s_val']
            cdf_fine = render_out['cdf_fine']
            gradient_error = render_out['gradient_error']
            weight_max = render_out['weight_max']
            weight_sum = render_out['weight_sum']

            # Loss
            colorm_error = (colorm_fine - true_rgb)
            colorm_fine_loss = (colorm_error-jt.zeros_like(colorm_error)).abs().sum() / mask_sum
            psnr = 20.0 * jt.log(1.0 / (((color_fine - true_rgb)**2).sum() / (mask_sum * 3.0)).sqrt())/math.log(10.0)

            eikonal_loss = gradient_error

            mask_loss = nn.bce_loss(weight_sum.clamp(1e-3, 1.0 - 1e-3), mask)

            loss = colorm_fine_loss +\
                   eikonal_loss * self.igr_weight +\
                   mask_loss * self.mask_weight

            self.optimizer.zero_grad()
            self.optimizer.backward(loss)
            self.optimizer.step()

            self.iter_step += 1

            # self.writer.add_scalar('Loss/loss', loss.numpy(), self.iter_step)
            # self.writer.add_scalar('Loss/color_loss', colorm_fine_loss.numpy(), self.iter_step)
            # self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss.numpy(), self.iter_step)
            # self.writer.add_scalar('Statistics/s_val', s_val.mean().numpy(), self.iter_step)
            # self.writer.add_scalar('Statistics/cdf',( (cdf_fine[:, :1] * mask).sum() / mask_sum).numpy(), self.iter_step)
            # self.writer.add_scalar('Statistics/weight_max',( (weight_max * mask).sum() / mask_sum).numpy(), self.iter_step)
            # self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)

            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {} lr={} psnr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr'], psnr))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                self.validate_image()

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh()

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()

    def get_image_perm(self):
        return jt.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = jt.load(str(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name)))
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        # self.color_network2.load_state_dict(checkpoint['color_network2_fine'])
        # self.pts_bias.load_state_dict(checkpoint['pts_bias'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            # 'color_network2_fine': self.color_network2.state_dict(),
            # 'pts_bias': self.pts_bias.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        jt.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d, rays_ld = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        rays_ld = rays_ld.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []

        for rays_o_batch, rays_d_batch, rays_ld_batch in zip(rays_o, rays_d, rays_ld):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = jt.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb,rays_ld=rays_ld_batch)

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('colorm_fine'):
                out_rgb_fine.append(render_out['colorm_fine'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}.jpg'.format(self.iter_step, i, idx)),
                           np.concatenate([img_fine[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]))
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.jpg'.format(self.iter_step, i, idx)),
                           normal_img[..., i])

    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d, rays_ld = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
        rays_ld = rays_ld.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch, rays_ld_batch in zip(rays_o, rays_d, rays_ld):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = jt.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb, rays_ld=rays_ld_batch)

            out_rgb_fine.append(render_out['colorm_fine'].detach().cpu().numpy())

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        return img_fine

    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0):
        bound_min = jt.array(self.dataset.object_bbox_min, dtype=jt.float32)
        bound_max = jt.array(self.dataset.object_bbox_max, dtype=jt.float32)

        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        # if world_space:
            # vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')

    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 40
        video_dir = os.path.join(self.base_exp_dir, 'render_inter2')
        os.makedirs(video_dir, exist_ok=True)
        modes = [0,40,80]
        for i in range(n_frames):
            print(i)
            images.append(self.render_novel_image(img_idx_0,
                                                  img_idx_1,
                                                  np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                          resolution_level=1))
            cv.imwrite(os.path.join(video_dir, '%06d.jpg'%(i+modes[2])), images[-1])
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        # fourcc = cv.VideoWriter_fourcc(*'mp4v')
        # h, w, _ = images[0].shape
        # writer = cv.VideoWriter(os.path.join(video_dir,'{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),
        #                         fourcc, 10, (w, h))

        # for image in images:
        #     writer.write(image)

        # writer.release()

    def render_sfs_images(self, resolution_level=4):
        # rays_o, rays_d = self.dataset.gen_rays_avg(resolution_level=resolution_level)
        
        os.makedirs(os.path.join(self.base_exp_dir, 'sfs_render'), exist_ok=True)
        for idx in range(self.dataset.n_images):
            print("rendering %dth image"%idx)
            # if idx<21:
            #     continue
            # if idx>3:
            #     break
            # rays_o, rays_d = self.dataset.gen_rays_avg(resolution_level=resolution_level)
            # rays_o, rays_d = self.dataset.gen_rays_at(0, resolution_level=resolution_level)
            # H, W, _ = rays_o.shape
            # rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
            # rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

            rays_o, rays_d, rays_ld = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
            H, W, _ = rays_o.shape
            rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
            rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
            rays_ld = rays_ld.reshape(-1, 3).split(self.batch_size)

            out_rgb_fine = []
            for rays_o_batch, rays_d_batch, rays_ld_batch in zip(rays_o, rays_d, rays_ld):
                near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)

                background_rgb =jt.ones([1, 3]) if self.use_white_bkgd else None

                render_out1 = self.renderer.render(rays_o_batch,
                                                rays_d_batch,
                                                near,
                                                far,
                                                cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                                background_rgb=background_rgb,
                                                rays_ld=rays_ld_batch)
                
                # render_out2 = self.renderer.render(rays_o2_batch,
                #                                 rays_d2_batch,
                #                                 near2,
                #                                 far2,
                #                                 cos_anneal_ratio=self.get_cos_anneal_ratio(),
                #                                 background_rgb=background_rgb,
                #                                 rays_d2 = rays_d2_batch)
                
                # colord = render_out1['colord_fine'].detach().cpu().numpy()
                # colors = render_out2['colors_fine'].detach().cpu().numpy()
                # color = 0.5*colord + 0.5*colors

                color = render_out1['colorm_fine'].detach().cpu().numpy()

                out_rgb_fine.append(color)

                del render_out1
                # del render_out2
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
            cv.imwrite(os.path.join(self.base_exp_dir, 'sfs_render', '%06d.jpg'%idx), img_fine)

if __name__ == '__main__':
    print('Hello Wooden')

    # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')

    args = parser.parse_args()

    # torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(world_space=True, resolution=512, threshold=args.mcube_threshold)
    elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
        _, img_idx_0, img_idx_1 = args.mode.split('_')
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0, img_idx_1)
    elif args.mode == 'caus':
        runner.render_sfs_images()

