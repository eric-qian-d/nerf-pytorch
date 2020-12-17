import os
from scipy.spatial.transform import Rotation as R
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2


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

def slerp(q0, q1, weights):
    dot_threshold = 1.9995
    weights = np.array(weights)
    q0 = np.array(q0)
    q1 = np.array(q1)
    dot = np.sum(q0 * q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > dot_threshold:
        result = q0[np.newaxis, :] + \
            weights[:, np.newaxis] * (q1 - q0)[np.newaxis, :]
        return (result.T / np.linalg.norm(result, axis=1)).T
    theta0 = np.arccos(dot)
    sin_theta0 = np.sin(theta0)
    theta = theta0 * weights
    sin_theta = np.sin(theta)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta0
    s1 = sin_theta / sin_theta0
    return (s0[:, np.newaxis] * q0[np.newaxis, :]) + \
        (s1[:, np.newaxis] * q1[np.newaxis, :])

def load_custom_data(basedir, half_res=False, testskip=1, inv=True):
    splits = ['train', 'test', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    novel_poses = []
    intrinsics = None
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            print('skip', skip)
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            if not inv:
              poses.append(np.array(frame['extrinsics']))
            else:
              poses.append(np.linalg.inv(np.array(frame['extrinsics'])))
            intrinsics = frame['intrinsics']
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)

        if s =='train':
            for i in range(len(poses) - 1):
                t1 = poses[i][:3,3:]
                r1 = R.from_matrix(poses[i][:3,:3])
                q1 = r1.as_quat()
                t2 = poses[i+1][:3,3:]
                r2 = R.from_matrix(poses[i+1][:3,:3])
                q2 = r2.as_quat()

                n = 2
                ws = np.linspace(0, 1, n, endpoint=False)

                for l, q in enumerate(slerp(q1, q2, ws)):
                    t = t1 * (1 - ws[l]) + t2 * ws[l]
                    r = R.from_quat(q).as_matrix()
                    extrinsics = np.eye(4)
                    extrinsics[:3,:4] = np.concatenate((r, t), axis=1)
                    novel_poses.append(extrinsics)

                    
               

        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    focal = intrinsics['fx'] * W
    
    #render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    render_poses = torch.tensor(novel_poses).float()
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 3))
        for i, img in enumerate(imgs):
            res = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            imgs_half_res[i] =  res
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        
    return imgs, poses, render_poses, [H, W, focal], i_split
