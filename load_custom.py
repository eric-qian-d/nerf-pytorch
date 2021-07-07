import os
from os.path import join
from glob import glob
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2
from image import image_resize

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_custom_data(scene, half_res=False, testskip=1, inv=True):
    splits = ['train', 'test', 'test', 'novel', 'full_original_track']
    data_dir = '/data/vision/billf/scratch/ericqian/neural-render/data/ibrnet/RealEstate10K-subset'

    all_imgs = []
    all_poses = []
    novel_poses = []
    intrinsics = None
    counts = [0]
    for s in splits:
        imgs = []
        poses = []
        if s == 'train' or s == 'test' or s == 'full_original_track':
            s_dir = join(data_dir, 'train')
            img_paths = glob(join(s_dir, 'frames', scene, '*'))
            img_paths.sort()
        elif s == 'novel':
            s_dir = join(data_dir, 'endpoint_interp')

        calibrations_path = join(s_dir, 'cameras', scene + '.txt')
        calibrations_file = open(calibrations_path)
        calibrations_info = calibrations_file.read()
        calibrations_lines  = calibrations_info.split('\n')
        calibrations_lines = calibrations_lines[1:] # don't need video link

        o = None
        for i, line in enumerate(calibrations_lines):
            # 90/10 train/test split
            if s == 'test' and i % 10 != 0:
                continue
            if s == 'train' and i % 10 == 0:
                continue

            items = line.split(' ')
            if len(items) < 19:
             break

            intrinsics = {'fx': float(items[1]), 'fy': float(items[2]), 'cx': float(items[3]), 'cy': float(items[4])}
            extrinsics = np.array(items[7:]).reshape((3, 4)).astype(np.float)

            w2c = np.eye(4)
            w2c[:3,:4] = extrinsics
            c2w = np.linalg.inv(w2c)
            m = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0],[0,0,0,1]])
            c2w = c2w @ m
            if o is None:
                o = c2w

            if s != 'novel':
                imgs.append(imageio.imread(img_paths[i]))
            poses.append(c2w)


        if s == 'novel':
            novel_poses = np.array(poses).astype(np.float32)
            continue
        if s == 'full_original_track':
            full_original_track_poses = np.array(poses).astype(np.float32)

        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)


        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    focal = intrinsics['fx'] * W
    
    render_poses = torch.tensor(novel_poses).float()
    full_original_track_poses = torch.tensor(full_original_track_poses).float()
    
    target_h = 128 if half_res else 256
    target_w = 256 if half_res else 512
    if target_h != H or target_w != W:
        W_original = W
        H = target_h
        W = target_w
        focal = focal/(W_original/W)

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 3))
        for i, img in enumerate(imgs):
            res = image_resize(img, width=W, height=H)
            imgs_half_res[i] =  res
        imgs = imgs_half_res
        
    return imgs, poses, render_poses, full_original_track_poses, [H, W, focal], i_split
