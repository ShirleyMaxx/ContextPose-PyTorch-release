import numpy as np
import torch

from mvn.utils.multiview import project_3d_points_to_image_plane_without_distortion
from mvn.utils.img import image_batch_to_torch

import os
import zipfile
import cv2

def make_collate_fn(randomize_n_views=True, min_n_views=10, max_n_views=31):

    def collate_fn(items):
        items = list(filter(lambda x: x is not None, items))
        if len(items) == 0:
            print("All items in batch are None")
            return None

        batch = dict()
        total_n_views = min(len(item['images']) for item in items)

        indexes = np.arange(total_n_views)
        if randomize_n_views:
            n_views = np.random.randint(min_n_views, min(total_n_views, max_n_views) + 1)
            indexes = np.random.choice(np.arange(total_n_views), size=n_views, replace=False)
        else:
            indexes = np.arange(total_n_views)

        batch['images'] = np.stack([np.stack([item['images'][i] for item in items], axis=0) for i in indexes], axis=0).swapaxes(0, 1)
        batch['detections'] = np.array([[item['detections'][i] for item in items] for i in indexes]).swapaxes(0, 1)
        batch['cameras'] = [[item['cameras'][i] for item in items] for i in indexes]

        batch['keypoints_3d'] = [item['keypoints_3d'] for item in items]
        # batch['cuboids'] = [item['cuboids'] for item in items]
        batch['indexes'] = [item['indexes'] for item in items]
        batch['subject'] = [item['subject'] for item in items]

        try:
            batch['pred_keypoints_3d'] = np.array([item['pred_keypoints_3d'] for item in items])
        except:
            pass

        return batch

    return collate_fn


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def prepare_batch(batch, device, config):
    # images
    images_batch = []
    for image_batch in batch['images']:
        image_batch = image_batch_to_torch(image_batch)
        image_batch = image_batch.to(device)
        images_batch.append(image_batch)

    images_batch = torch.stack(images_batch, dim=0)

    # 3D keypoints
    keypoints_3d_batch_gt = torch.from_numpy(np.stack(batch['keypoints_3d'], axis=0)[:, :, :3]).float().to(device)      # (b, n_joints, 3)

    # keypoints validity
    keypoints_validity_batch_gt = torch.from_numpy(np.stack(batch['keypoints_3d'], axis=0)[:, :, 3:]).float().to(device)     # (b, n_joints, 1)

    # projection matricies
    proj_matricies_batch = torch.stack([torch.stack([torch.from_numpy(camera.projection) for camera in camera_batch], dim=0) for camera_batch in batch['cameras']], dim=0).transpose(1, 0)  # shape (batch_size, n_views, 3, 4)
    proj_matricies_batch = proj_matricies_batch.float().to(device)


    return images_batch, keypoints_3d_batch_gt, keypoints_validity_batch_gt, proj_matricies_batch

_im_zfile = []

def zipreader_imread(filename, flags=cv2.IMREAD_COLOR):
    global _im_zfile
    path = filename
    pos_at = path.index('@')
    if pos_at == -1:
        print("character '@' is not found from the given path '%s'" % (path))
        assert 0
    path_zip = path[0:pos_at]
    if not os.path.isfile(path_zip):
        print("zip file '%s' is not found" % (path_zip))
        assert 0
    for i in range(len(_im_zfile)):
        if _im_zfile[i]['path'] == path_zip:
            path_img = os.path.join(_im_zfile[i]['zipfile'].namelist()[0], path[pos_at+2:])
            data = _im_zfile[i]['zipfile'].read(path_img)
            return cv2.imdecode(np.frombuffer(data, np.uint8), flags)

    _im_zfile.append({
        'path': path_zip,
        'zipfile': zipfile.ZipFile(path_zip, 'r')
    })
    path_img = os.path.join(_im_zfile[-1]['zipfile'].namelist()[0], path[pos_at+2:])
    data = _im_zfile[-1]['zipfile'].read(path_img)

    return cv2.imdecode(np.frombuffer(data, np.uint8), flags)