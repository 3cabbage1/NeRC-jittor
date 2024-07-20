import cv2 as cv
import jittor as jt
from jittor import contrib
import numpy as np
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import math
import random
# from PIL import Image, ImageOps, ImageEnhance, PILLOW_VERSION

try:
    import accimage
except ImportError:
    accimage = None
import scipy.stats as st
import numbers
import types
import collections
# import matplotlib.pyplot as plt
# import torchvision.transforms as transforms
from scipy.signal import convolve2d
import json
import imageio

np.random.seed(0)


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose =np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]
    return intrinsics, pose


def ReflectionSythesis(B_, R_, kernel_sizes=11, low_sigma=2, high_sigma=5, low_gamma=1.3, high_gamma=1.3):
    """Reflection image data synthesis for weakly-supervised learning
    of ICCV 2017 paper *"A Generic Deep Architecture for Single Image Reflection Removal and Image Smoothing"*
    """

    kernel_size = np.random.choice(kernel_sizes)
    sigma = np.random.uniform(low_sigma, high_sigma)
    gamma = np.random.uniform(low_gamma, high_gamma)

    R_blur = R_
    kernel = cv.getGaussianKernel(11, sigma)
    kernel2d = np.dot(kernel, kernel.T)

    for i in range(3):
        R_blur[..., i] = convolve2d(R_blur[..., i], kernel2d, mode='same')

    M_ = B_ + R_blur

    if np.max(M_) > 1:
        m = M_[M_ > 1]
        m = (np.mean(m) - 1) * gamma
        R_blur = np.clip(R_blur - m, 0, 1)
        M_ = np.clip(R_blur + B_, 0, 1)

    return M_


def get_tf_cams(cam_dict, target_radius=1.):
    cam_centers = []
    for frame in cam_dict['frames']:
        C2W = np.array(frame['transform_matrix']).reshape((4, 4))
        cam_centers.append(C2W[:3, 3:4])

    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center
    scale = target_radius / radius

    return translate, scale


def load_blender_data(posedir):
    metas = {}
    with open(posedir, 'r') as fp:
        metas = json.load(fp)
    meta = metas
    poses = {}
    lds = {}
    cats = {}
    cates = {"Camera": 1.0, "Light": 2.0, "Both": 3.0}
    translate, scale = get_tf_cams(meta, target_radius=1.)
    for frame in meta['frames']:
        fname = frame['file_path'].split('/')[-1]

        pt = np.array(frame['transform_matrix']).reshape((4, 4))
        cam_center = pt[:3, 3]
        cam_center = (cam_center + translate) * scale
        pt[:3, 3] = cam_center
        poses[fname] = pt

        ld = np.array(frame['light_location']).reshape((3, 1))
        lds[fname] = (ld[:, 0] + translate) * scale

        cats[fname] = cates[frame['move']]
        if cats[fname] == 2.0:
            poses[fname][:3, 3] = lds[fname]

    H, W = [800, 800]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    K = np.array([[focal, 0.0, W * 1.0 / 2, 0.0],
                  [0.0, focal, H * 1.0 / 2, 0.0],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0]]).reshape(4, 4).astype(np.float32)

    return K, poses, lds, cats


def convert_pose(C2W):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W


def align(p, g):
    a, b = np.polyfit(np.squeeze(p), np.squeeze(g), deg=1)
    pred_metric = a * p + b
    return pred_metric


class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        # self.device ='cuda'
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        camera_dict = os.path.join(self.data_dir, self.render_cameras_name)

        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'train\*.png')))
        self.n_images = len(self.images_lis)
        self.images_lisR = self.images_lis
        self.n_imagesR = len(self.images_lis)
        self.images_lisM = self.images_lis
        self.n_imagesM = len(self.images_lis)

        # self.images_npM = np.stack([cv.imread(im_name) for im_name in self.images_lisM]) / 255.0
        # alpha = self.images_npM[...,-1:]
        # self.images_npM = self.images_npM[...,:3]*alpha + (1-alpha)
        self.masks_np = np.stack([np.ones_like(cv.imread(im_name)[:, :, 0]) for im_name in self.images_lis])

        K, poses, lds, cats = load_blender_data(camera_dict)
        # print(K)''
        # print(poses)''
        # print(lds)''
        # print(cats)''

        self.intrinsics_all = []
        self.ld_all = []
        self.pose_all = []
        self.images_npM = []
        self.images_npM0 = []
        count = 0
        for i in range(self.n_images):
            ''''''
            img_name = self.images_lis[i].split('\\')[-1]
            if cats[img_name] != 3.0:
                continue

            img = cv.imread(self.images_lisM[i]).astype(np.float32) / 255.0
            self.images_npM.append(jt.array(img).float32())
            self.images_npM0.append((img * 255.0).astype(np.uint8))

            C2W = poses[img_name].astype(np.float32)
            C2W = convert_pose(C2W)
            ld = lds[img_name][:, None].astype(np.float32)
            ld[1, 0] = -ld[1, 0]
            ld[2, 0] = -ld[2, 0]

            W2C = np.linalg.inv(C2W)
            P = K @ W2C
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)

            count += 1

            self.intrinsics_all.append(jt.array(intrinsics).float32())
            self.ld_all.append(jt.array(ld).float32())
            self.pose_all.append(jt.array(pose).float32())

        self.n_images = count
        # jt.flags.use_cuda=0#在cpu运行
        self.imagesM = jt.stack(self.images_npM, dim=0)

        self.masks = jt.array(self.masks_np.astype(np.float32))  # [n_images, H, W, 3]

        # if self.device=='cpu':
        #     jt.flags.use_cuda = 0
        # else:
        #     jt.flags.use_cuda = 2



        self.intrinsics_all = jt.stack(self.intrinsics_all) # [n_images, 4, 4]
        self.ld_all = jt.stack(self.ld_all) # [n_images, 1, 3]


        self.intrinsics_all_inv =jt.linalg.inv(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = jt.stack(self.pose_all)  # [n_images, 4, 4]
        self.H, self.W = self.imagesM.shape[1], self.imagesM.shape[2]
        self.image_pixels = self.H * self.W

        def normalize(x):
            return x / np.linalg.norm(x)

        def viewmatrix(z, up, pos):
            vec2 = normalize(z)
            vec1_avg = up
            vec0 = normalize(np.cross(vec1_avg, vec2))
            vec1 = normalize(np.cross(vec2, vec0))
            m = np.stack([vec0, vec1, vec2, pos], 1)
            return m

        def poses_avg(poses):
            hwf = poses[0, :3, -1:]

            center = poses[:, :3, 3].mean(0)
            vec2 = normalize(poses[:, :3, 2].sum(0))
            up = poses[:, :3, 1].sum(0)
            c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

            bottom = np.reshape([0, 0, 0, 1.], [1, 4])
            c2w = np.concatenate([c2w[:3, :4], bottom], -2)

            return c2w

        poses = self.pose_all.detach().cpu().numpy()
        self.pose_a = poses_avg(poses)
        self.pose_a = jt.array(self.pose_a.astype(np.float32))

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([1.01, 1.01, 1.01, 1.0])

        self.object_bbox_min = object_bbox_min[:3]
        self.object_bbox_max = object_bbox_max[:3]
        print('Load data: End')

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = jt.linspace(0, self.W - 1, self.W // l)
        ty = jt.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = jt.meshgrid(tx, ty)
        p = jt.stack([pixels_x, pixels_y, jt.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = jt.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / jt.norm(p, p=2, dim=p.dim-1, keepdims=True)  # W, H, 3
        rays_v = jt.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        rays_ld = self.ld_all[img_idx, None, None, :, 0].expand(rays_v.shape)

        return rays_o.transpose(0, 1), rays_v.transpose(0, 1), rays_ld

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = jt.randint(low=0, high=self.W, shape=(batch_size,))
        pixels_y = jt.randint(low=0, high=self.H, shape=(batch_size,))
        color = self.imagesM[img_idx,pixels_y, pixels_x]  # batch_size, 3
        mask = self.masks[img_idx,pixels_y, pixels_x]  # batch_size, 3

        p = jt.stack([pixels_x, pixels_y,jt.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = jt.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()  # batch_size, 3
        rays_v = p / jt.norm(p, p=2, dim=-1, keepdims=True)  # batch_size, 3
        rays_v = jt.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, :3, 3].expand(rays_v.shape)  # batch_size, 3
        rays_ld = self.ld_all[img_idx, :, 0].expand(rays_v.shape)
        return jt.concat([jt.array(rays_o), jt.array(rays_v), jt.array(color), jt.array(mask.unsqueeze(-1)), jt.array(rays_ld)], dim=-1)  # batch_size, 10

    def gen_rays_avg(self, resolution_level=1):
        """
        Interpolate pose in the median position.
        """
        l = resolution_level
        tx = jt.linspace(0, self.W - 1, self.W // l)
        ty = jt.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = jt.meshgrid(tx, ty)

        p = jt.stack([pixels_x, pixels_y, jt.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = jt.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / jt.norm(p, p=2, dim=p.dim-1, keepdims=True)  # W, H, 3
        rays_v = jt.matmul(self.pose_a[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_a[None, None, :3, 3].expand(rays_v.shape)  # W, H, 3

        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = jt.linspace(0, self.W - 1, self.W // l)
        ty = jt.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = jt.meshgrid(tx, ty)
        p = jt.stack([pixels_x, pixels_y, jt.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = jt.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / jt.norm(p, p=2, dim=p.dim-1, keepdims=True)  # W, H, 3

        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio

        lds = self.ld_all[idx_0, :3, 0] * (1.0 - ratio) + self.ld_all[idx_1, :3, 0] * ratio

        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = jt.array(pose[:3, :3]).cuda()
        trans =jt.array(pose[:3, 3]).cuda()
        rays_v = jt.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        rays_ld = lds[None, None, :3].expand(rays_v.shape)  # W, H, 3

        return rays_o.transpose(0, 1), rays_v.transpose(0, 1), rays_ld

    def near_far_from_sphere(self, rays_o, rays_d):
        a = jt.sum(rays_d ** 2,dim=-1,  keepdims=True)
        b = 2.0 * jt.sum(rays_o * rays_d,dim=-1,  keepdims=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        img = self.images_npM0[idx]
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)

