import os
import shutil
import argparse
import time
import json
from datetime import datetime
from collections import defaultdict
from itertools import islice
import pickle
import copy
from tqdm import tqdm
import h5py
from PIL import Image
import numpy as np
np.set_printoptions(suppress=True)
import cv2
import prettytable

import torch
from torch import nn
from torch import autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from tensorboardX import SummaryWriter

from mvn.models.triangulation import VolumetricTriangulationNet
from mvn.models.loss import KeypointsMSELoss, KeypointsMSESmoothLoss, KeypointsMAELoss, KeypointsL2Loss, VolumetricCELoss, LimbLengthError

from mvn.utils import img, multiview, op, vis, misc, cfg
from mvn.utils.cfg import config, update_config, update_dir
from mvn import datasets
from mvn.datasets import utils as dataset_utils
from mvn.utils.vis import JOINT_NAMES_DICT


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True, help="Path, where config file is stored")
    parser.add_argument('--eval', action='store_true', help="If set, then only evaluation will be done")
    parser.add_argument('--eval_dataset', type=str, default='val', help="Dataset split on which evaluate. Can be 'train' and 'val'")

    parser.add_argument("--local_rank", type=int, help="Local rank of the process on the node")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--sync_bn', action='store_true', help="If set, then utilize pytorch convert_syncbn_model")

    parser.add_argument("--logdir", type=str, default="logs/", help="Path, where logs will be stored")
    parser.add_argument("--azureroot", type=str, default="", help="Root path, where codes are stored")

    args = parser.parse_args()
    # update config
    update_config(args.config)
    update_dir(args.azureroot, args.logdir)
    return args


def setup_human36m_dataloaders(config, is_train, distributed_train, rank = None, world_size = None):
    train_dataloader = None
    if is_train:
        # train
        train_dataset = eval('datasets.' + config.dataset.train_dataset)(
            root=config.dataset.root,
            pred_results_path=config.train.pred_results_path,
            train=True,
            test=False,
            image_shape=config.model.image_shape,
            labels_path=config.dataset.train_labels_path,
            with_damaged_actions=config.train.with_damaged_actions,
            scale_bbox=config.train.scale_bbox,
            kind=config.kind,
            undistort_images=config.train.undistort_images,
            ignore_cameras=config.train.ignore_cameras,
            crop=config.train.crop,
            erase=config.train.erase,
            data_format=config.dataset.data_format
        )

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed_train else None

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.train.batch_size,
            shuffle=config.train.shuffle and (train_sampler is None), # debatable
            sampler=train_sampler,
            collate_fn=dataset_utils.make_collate_fn(randomize_n_views=config.train.randomize_n_views,
                                                     min_n_views=config.train.min_n_views,
                                                     max_n_views=config.train.max_n_views),
            num_workers=config.train.num_workers,
            worker_init_fn=dataset_utils.worker_init_fn,
            pin_memory=True
        )

    # val
    val_dataset = eval('datasets.' + config.dataset.val_dataset)(
        root=config.dataset.root,
        pred_results_path=config.val.pred_results_path,
        train=False,
        test=True,
        image_shape=config.model.image_shape,
        labels_path=config.dataset.val_labels_path,
        with_damaged_actions=config.val.with_damaged_actions,
        retain_every_n_frames_in_test=config.val.retain_every_n_frames_in_test,
        scale_bbox=config.val.scale_bbox,
        kind=config.kind,
        undistort_images=config.val.undistort_images,
        ignore_cameras=config.val.ignore_cameras,
        crop=config.val.crop,
        erase=config.val.erase,
        rank=rank,
        world_size=world_size,
        data_format=config.dataset.data_format
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.val.batch_size,
        shuffle=config.val.shuffle,
        collate_fn=dataset_utils.make_collate_fn(randomize_n_views=config.val.randomize_n_views,
                                                 min_n_views=config.val.min_n_views,
                                                 max_n_views=config.val.max_n_views),
        num_workers=config.val.num_workers,
        worker_init_fn=dataset_utils.worker_init_fn,
        pin_memory=True
    )

    return train_dataloader, val_dataloader, train_sampler, val_dataset.dist_size


def setup_dataloaders(config, is_train=True, distributed_train=False, rank = None, world_size=None):
    if config.dataset.kind == 'human36m':
        train_dataloader, val_dataloader, train_sampler, dist_size = setup_human36m_dataloaders(config, is_train, distributed_train, rank, world_size)
        _, whole_val_dataloader, _, _ = setup_human36m_dataloaders(config, is_train, distributed_train)
    else:
        raise NotImplementedError("Unknown dataset: {}".format(config.dataset.kind))
    
    return train_dataloader, val_dataloader, train_sampler, whole_val_dataloader, dist_size


def setup_experiment(config, model_name, is_train=True):
    prefix = "" if is_train else "eval_"

    if config.title:
        experiment_title = config.title + "_" + model_name
    else:
        experiment_title = model_name

    experiment_title = prefix + experiment_title

    experiment_name = '{}@{}'.format(experiment_title, datetime.now().strftime("%d.%m.%Y-%H:%M:%S"))
    print("Experiment name: {}".format(experiment_name))

    experiment_dir = os.path.join(config.logdir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    shutil.copy(args.config, os.path.join(experiment_dir, "config.yaml"))

    # tensorboard
    writer = SummaryWriter(os.path.join(experiment_dir, "tb"))

    # dump config to tensorboard
    writer.add_text(misc.config_to_str(config), "config", 0)

    return experiment_dir, writer


def one_epoch_full(model, criterion, opt_dict, config, dataloader, device, epoch, n_iters_total=0, is_train=True, lr=None, mean_and_std=None, limb_length = None, caption='', master=False, experiment_dir=None, writer=None, whole_val_dataloader=None, dist_size=None):
    name = "train" if is_train else "val"
    model_type = config.model.name

    if is_train:
        if config.model.backbone.fix_weights:
            model.module.backbone.eval()
            if config.model.volume_net.use_feature_v2v:
                model.module.process_features.train()
            model.module.volume_net.train()
        else:
            model.train()
    else:
        model.eval()

    metric_dict = defaultdict(list)

    results = defaultdict(list)


    # used to turn on/off gradients
    grad_context = torch.autograd.enable_grad if is_train else torch.no_grad
    with grad_context():
        end = time.time()

        if master:
            if is_train and config.train.n_iters_per_epoch is not None:
                pbar = tqdm(total=min(config.train.n_iters_per_epoch, len(dataloader)))
            else:
                pbar = tqdm(total=len(dataloader))

        iterator = enumerate(dataloader)
        if is_train and config.train.n_iters_per_epoch is not None:
            iterator = islice(iterator, config.train.n_iters_per_epoch)

        for iter_i, batch in iterator:
            # measure data loading time
            data_time = time.time() - end

            if batch is None:
                print("Found None batch")
                continue

            images_batch, keypoints_3d_gt, keypoints_validity_gt, proj_matricies_batch = dataset_utils.prepare_batch(batch, device, config)

            keypoints_2d_pred, cuboids_pred, base_points_pred = None, None, None
            if model_type == "vol":
                voxel_keypoints_3d_pred, keypoints_3d_pred, heatmaps_pred,\
                    volumes_pred, ga_mask_gt, atten_global, confidences_pred, cuboids_pred, coord_volumes_pred, base_points_pred =\
                    model(images_batch, proj_matricies_batch, batch, keypoints_3d_gt)

            batch_size, n_views, image_shape = images_batch.shape[0], images_batch.shape[1], tuple(images_batch.shape[3:])
            n_joints = keypoints_3d_pred.shape[1]

            keypoints_binary_validity_gt = (keypoints_validity_gt > 0.0).type(torch.float32)

            scale_keypoints_3d = config.loss.scale_keypoints_3d

            # calculate loss
            total_loss = 0.0
            loss = criterion(keypoints_3d_pred * scale_keypoints_3d, keypoints_3d_gt * scale_keypoints_3d, keypoints_binary_validity_gt)
            total_loss += loss
            metric_dict[config.loss.criterion].append(loss.item())

            # volumetric ce loss
            if config.loss.use_volumetric_ce_loss :
                volumetric_ce_criterion = VolumetricCELoss()

                loss = volumetric_ce_criterion(coord_volumes_pred, volumes_pred, keypoints_3d_gt, keypoints_binary_validity_gt)
                metric_dict['volumetric_ce_loss'].append(loss.item())

                total_loss += config.loss.volumetric_ce_loss_weight * loss

            # global attention (3D heatmap) loss
            if config.loss.use_global_attention_loss:
                loss = nn.MSELoss(reduction='mean')(ga_mask_gt, atten_global)
                metric_dict['global_attention_loss'].append(loss.item())
                total_loss += config.loss.global_attention_loss_weight * loss

            metric_dict['total_loss'].append(total_loss.item())
            metric_dict['limb_length_error'].append(LimbLengthError()(keypoints_3d_pred.detach(), keypoints_3d_gt))

            if is_train:
                if not torch.isnan(total_loss):
                    for key in opt_dict.keys():
                        opt_dict[key].zero_grad()
                    total_loss.backward()

                    if config.loss.grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.loss.grad_clip / config.train.volume_net_lr)

                    metric_dict['grad_norm_times_volume_net_lr'].append(config.train.volume_net_lr * misc.calc_gradient_norm(filter(lambda x: x[1].requires_grad, model.named_parameters())))
                    if lr is not None:
                        for key in lr.keys():
                            metric_dict['lr_{}'.format(key)].append(lr[key])

                    for key in opt_dict.keys():
                        opt_dict[key].step()

            # calculate metrics
            l2 = KeypointsL2Loss()(keypoints_3d_pred * scale_keypoints_3d, keypoints_3d_gt * scale_keypoints_3d, keypoints_binary_validity_gt)
            metric_dict['l2'].append(l2.item())

            # base point l2
            if base_points_pred is not None:
                base_point_l2_list = []
                for batch_i in range(batch_size):
                    base_point_pred = base_points_pred[batch_i]

                    if config.model.kind == "coco":
                        base_point_gt = (keypoints_3d_gt[batch_i, 11, :3] + keypoints_3d[batch_i, 12, :3]) / 2
                    elif config.model.kind == "mpii":
                        base_point_gt = keypoints_3d_gt[batch_i, 6, :3]

                    base_point_l2_list.append(torch.sqrt(torch.sum((base_point_pred * scale_keypoints_3d - base_point_gt * scale_keypoints_3d) ** 2)).item())

                base_point_l2 = 0.0 if len(base_point_l2_list) == 0 else np.mean(base_point_l2_list)
                metric_dict['base_point_l2'].append(base_point_l2)

            # save answers for evalulation
            if not is_train:
                results['keypoints_gt'].append(keypoints_3d_gt.detach().cpu().numpy())    # (b, 17, 3)
                results['keypoints_3d'].append(keypoints_3d_pred.detach().cpu().numpy())    # (b, 17, 3)
                results['proj_matricies_batch'].append(proj_matricies_batch.detach().cpu().numpy())  #(b, n_view, 3,4)
                results['indexes'].append(batch['indexes'])


            # plot visualization
            if master:
                if config.batch_output:  
                    if n_iters_total % config.vis_freq == 0:# or total_l2.item() > 500.0:
                        sample_i = iter_i*config.vis_freq + n_iters_total
                        vis_kind = config.kind
                        if config.dataset.transfer_cmu_to_human36m:
                            vis_kind = "coco"

                        for batch_i in range(min(batch_size, config.vis_n_elements)):
                            keypoints_vis = vis.visualize_batch(
                                images_batch, heatmaps_pred, keypoints_2d_pred, proj_matricies_batch,
                                keypoints_3d_gt, keypoints_3d_pred,
                                kind=vis_kind,
                                cuboids_batch=cuboids_pred,
                                confidences_batch=confidences_pred,
                                batch_index=batch_i, size=5,
                                max_n_cols=10
                            )
                            writer.add_image("{}/keypoints_vis/{}".format(name, batch_i), keypoints_vis.transpose(2, 0, 1), global_step=n_iters_total)

                            heatmaps_vis = vis.visualize_heatmaps(
                                images_batch, heatmaps_pred,
                                kind=vis_kind,
                                batch_index=batch_i, size=5,
                                max_n_rows=10, max_n_cols=18
                            )
                            writer.add_image("{}/heatmaps/{}".format(name, batch_i), heatmaps_vis.transpose(2, 0, 1), global_step=n_iters_total)

                            if model_type == "vol":
                                volumes_vis = vis.visualize_volumes(
                                    images_batch, volumes_pred, proj_matricies_batch,
                                    kind=vis_kind,
                                    cuboids_batch=cuboids_pred,
                                    batch_index=batch_i, size=5,
                                    max_n_rows=1, max_n_cols=18
                                )
                                writer.add_image("{}/volumes/{}".format(name, batch_i), volumes_vis.transpose(2, 0, 1), global_step=n_iters_total)


                    # dump weights to tensoboard
                    if n_iters_total % config.vis_freq == 0:
                        for p_name, p in model.named_parameters():
                            try:
                                writer.add_histogram(p_name, p.clone().cpu().data.numpy(), n_iters_total)
                            except ValueError as e:
                                print(e)
                                print(p_name, p)
                                exit()

                    # dump to tensorboard per-iter loss/metric stats
                    if is_train:
                        for title, value in metric_dict.items():
                            writer.add_scalar("{}/{}".format(name, title), value[-1], n_iters_total)

                    # measure elapsed time
                    batch_time = time.time() - end
                    end = time.time()

                    # dump to tensorboard per-iter time stats
                    writer.add_scalar("{}/batch_time".format(name), batch_time, n_iters_total)
                    writer.add_scalar("{}/data_time".format(name), data_time, n_iters_total)

                    # dump to tensorboard per-iter stats about sizes
                    writer.add_scalar("{}/batch_size".format(name), batch_size, n_iters_total)
                    writer.add_scalar("{}/n_views".format(name), n_views, n_iters_total)

                n_iters_total += 1
                pbar.update(1)

    # calculate evaluation metrics
    if not is_train:
        if dist_size is not None:
            term_list = ['keypoints_gt', 'keypoints_3d', 'proj_matricies_batch', 'indexes']
            for term in term_list:
                results[term] = np.concatenate(results[term])
                buffer = [torch.zeros(dist_size[-1], *results[term].shape[1:]).cuda() for i in range(len(dist_size))]
                scatter_tensor = torch.zeros_like(buffer[0])
                scatter_tensor[:results[term].shape[0]] = torch.tensor(results[term]).cuda()
                torch.distributed.all_gather(buffer, scatter_tensor)
                results[term] = torch.cat([tensor[:n] for tensor, n in zip(buffer, dist_size)], dim = 0).cpu().numpy()

    if master:
        if not is_train:
            try:
                if dist_size is None:
                    print('evaluating....')
                    scalar_metric, full_metric = dataloader.dataset.evaluate(results['keypoints_gt'], results['keypoints_3d'], results['proj_matricies_batch'], config)
                else:
                    scalar_metric, full_metric = whole_val_dataloader.dataset.evaluate(results['keypoints_gt'], results['keypoints_3d'], results['proj_matricies_batch'], config)
            except Exception as e:
                print("Failed to evaluate. Reason: ", e)
                scalar_metric, full_metric = 0.0, {}

            metric_dict['dataset_metric'].append(scalar_metric)
            metric_dict['limb_length_error'] = [LimbLengthError()(results['keypoints_3d'], results['keypoints_gt'])]

            checkpoint_dir = os.path.join(experiment_dir, "checkpoints", "{:04}".format(epoch))
            os.makedirs(checkpoint_dir, exist_ok=True)

            # dump results
            with open(os.path.join(checkpoint_dir, "results.pkl"), 'wb') as fout:
                pickle.dump(results, fout)

            # dump full metric
            with open(os.path.join(checkpoint_dir, "metric.json".format(epoch)), 'w') as fout:
                json.dump(full_metric, fout, indent=4, sort_keys=True)

        # dump to tensorboard per-epoch stats
        for title, value in metric_dict.items():
            writer.add_scalar("{}/{}_epoch".format(name, title), np.mean(value), epoch)

    return n_iters_total


def init_distributed(args):
    if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) < 1:
        return False

    torch.cuda.set_device(args.local_rank)

    assert os.environ["MASTER_PORT"], "set the MASTER_PORT variable or use pytorch launcher"
    assert os.environ["RANK"], "use pytorch launcher and explicityly state the rank of the process"

    torch.manual_seed(args.seed)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")

    return True


def main(args):
    print("Number of available GPUs: {}".format(torch.cuda.device_count()))

    is_distributed = init_distributed(args)
    master = True
    if is_distributed and os.environ["RANK"]:
        master = int(os.environ["RANK"]) == 0
        rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    else:
        rank = world_size = None

    if is_distributed:
        device = torch.device(args.local_rank)
    else:
        device = torch.device(0)

    config.train.n_iters_per_epoch = config.train.n_objects_per_epoch // config.train.batch_size                        

    model = {
        "vol": VolumetricTriangulationNet
    }[config.model.name](config, device)

    # experiment
    experiment_dir, writer = None, None
    if master:
        experiment_dir, writer = setup_experiment(config, type(model).__name__, is_train=not args.eval)
        shutil.copy('mvn/models/v2v_net.py', experiment_dir)

    if config.model.init_weights:
        checkpoint_path = None
        if config.model.checkpoint != None:
            checkpoint_path = config.model.checkpoint
        elif os.path.isfile(os.path.join(config.logdir, "resume_weights_path.pth")):
            checkpoint_path = torch.load(os.path.join(config.logdir, "resume_weights_path.pth"))
        if checkpoint_path != None and os.path.isfile(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=device)
            for key in list(state_dict.keys()):
                new_key = key.replace("module.", "")
                state_dict[new_key] = state_dict.pop(key)
            try:
                model.load_state_dict(state_dict, strict=True)
            except:
                print('Warning: Final layer do not match!')
                for key in list(state_dict.keys()):
                    if 'final_layer' in key:
                        state_dict.pop(key)
                model.load_state_dict(state_dict, strict=True)
            print("Successfully loaded weights for {} model from {}".format(config.model.name, checkpoint_path))
            del state_dict
        else:
            print("Failed loading weights for {} model as no checkpoint found at {}".format(config.model.name, checkpoint_path))

    # sync bn in multi-gpus
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)

    # criterion
    criterion_class = {
        "MSE": KeypointsMSELoss,
        "MSESmooth": KeypointsMSESmoothLoss,
        "MAE": KeypointsMAELoss
    }[config.loss.criterion]

    if config.loss.criterion == "MSESmooth":
        criterion = criterion_class(config.loss.mse_smooth_threshold).to(device)
    else:
        criterion = criterion_class().to(device)

    # optimizer
    opt_dict = None
    lr_schd_dict = None
    lr_dict = None
    if not args.eval:
        if config.model.name == "vol":
            opt_dict = {}
            lr_schd_dict = {}
            lr_dict = {}
            # backbone opt
            if not config.model.backbone.fix_weights:
                params_2d = [{'params': model.backbone.parameters(), 'lr': config.train.backbone_lr}]
                opt_2d = optim.Adam(params_2d)
                lr_schd_2d = optim.lr_scheduler.MultiStepLR(opt_2d, config.train.backbone_lr_step, config.train.backbone_lr_factor)
                opt_dict.update({'2d': opt_2d})
                lr_schd_dict.update({'2d': lr_schd_2d})
                lr_dict.update({'2d': config.train.backbone_lr})
            # volume_net opt
            params_3d = [{'params': model.volume_net.parameters(), 'lr': config.train.volume_net_lr}]
            if config.model.volume_net.use_feature_v2v:
                params_3d.append({'params': model.process_features.parameters(), 'lr': config.train.process_features_lr})
            opt_3d = optim.Adam(params_3d)
            lr_schd_3d = optim.lr_scheduler.MultiStepLR(opt_3d, config.train.volume_net_lr_step, config.train.volume_net_lr_factor)
            opt_dict.update({'3d': opt_3d})
            lr_schd_dict.update({'3d': lr_schd_3d})
            lr_dict.update({'3d': config.train.volume_net_lr})
        else:
            assert 0, "Only support vol optimizer."
        # load optimizer if has
        if config.model.init_weights and checkpoint_path != None:
            optimizer_path = checkpoint_path.replace('weights', 'optimizer')
            if os.path.isfile(optimizer_path):
                try:
                    optimizer_dict = torch.load(optimizer_path, map_location=device)
                    if config.model.name == 'vol':
                        opt_dict['3d'].load_state_dict(optimizer_dict['optimizer_3d'])
                        lr_schd_dict['3d'].load_state_dict(optimizer_dict['scheduler_3d'])
                        if 'scheduler_2d' in optimizer_dict.keys():
                            opt_dict['2d'].load_state_dict(optimizer_dict['optimizer_2d'])
                            lr_schd_dict['2d'].load_state_dict(optimizer_dict['scheduler_2d'])
                    else:
                        assert 0, "Only support vol optimizer."
                    del optimizer_dict
                    print("Successfully loaded optimizer parameters for {} model".format(config.model.name))
                except:
                    print("Warning: optimizer does not match! Failed loading optimizer parameters for {} model".format(config.model.name))
            else:
                print("Failed loading optimizer parameters for {} model as no optimizer found at {}".format(config.model.name, optimizer_path))

    # datasets
    print("Loading data...")
    train_dataloader, val_dataloader, train_sampler, whole_val_dataloader, dist_size = setup_dataloaders(config, distributed_train=is_distributed, rank=rank, world_size=world_size)

    if config.model.name == 'vol':
        print("Loading limb length mean & std...")
        mean_and_std = {}
        print(config.train.limb_length_path)
        limb_length_file = h5py.File(config.train.limb_length_path, 'r')
        mean = torch.from_numpy(np.array(limb_length_file['mean'])).float().cuda()
        std = torch.from_numpy(np.array(limb_length_file['std'])).float().cuda()
        limb_length = {'mean': mean[:-1], 'std': std[:-1]}
        mean_and_std = limb_length

    # multi-gpu
    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[device], output_device=args.local_rank)

    if not args.eval:
        # train loop
        n_iters_total_train, n_iters_total_val = 0, 0
        
        for epoch in range(config.train.n_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            if config.model.name == 'vol':
                n_iters_total_train = one_epoch_full(model, criterion, opt_dict, config, train_dataloader, device, epoch, n_iters_total=n_iters_total_train, is_train=True, lr=lr_dict, mean_and_std=mean_and_std, limb_length=limb_length, master=master, experiment_dir=experiment_dir, writer=writer)
                n_iters_total_val = one_epoch_full(model, criterion, opt_dict, config, val_dataloader, device, epoch, n_iters_total=n_iters_total_val, is_train=False, mean_and_std=mean_and_std, limb_length=limb_length, master=master, experiment_dir=experiment_dir, writer=writer, whole_val_dataloader=whole_val_dataloader, dist_size=dist_size)
                for key in lr_schd_dict.keys():
                    lr_schd_dict[key].step()
                    try:
                        lr_dict[key] = lr_schd_dict[key].get_last_lr()[0]
                    except: # old PyTorch
                        lr_dict[key] = lr_schd_dict[key].get_lr()[0]
            else:
                assert 0, "only support training vol model."

            if master:
                checkpoint_dir = os.path.join(experiment_dir, "checkpoints", "{:04}".format(epoch))
                os.makedirs(checkpoint_dir, exist_ok=True)

                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "weights.pth"))
                torch.save(os.path.join(checkpoint_dir, "weights.pth"), os.path.join(config.logdir, "resume_weights_path.pth"))
                if config.model.name == 'vol':
                    if config.model.backbone.fix_weights:
                        torch.save({'optimizer_3d': opt_dict['3d'].state_dict(), \
                                    'scheduler_3d': lr_schd_dict['3d'].state_dict()}, \
                                    os.path.join(checkpoint_dir, "optimizer.pth"))
                    else:
                        torch.save({'optimizer_2d': opt_dict['2d'].state_dict(), \
                                    'optimizer_3d': opt_dict['3d'].state_dict(), \
                                    'scheduler_2d': lr_schd_dict['2d'].state_dict(), \
                                    'scheduler_3d': lr_schd_dict['3d'].state_dict()}, \
                                    os.path.join(checkpoint_dir, "optimizer.pth"))
                else:
                    assert 0, "only support saving vol model."
            print("{} iters done.".format(n_iters_total_train))
    else:
        dataloader = train_dataloader if args.eval_dataset == 'train' else val_dataloader

        one_epoch_full(model, criterion, opt_dict, config, dataloader, device, 0, n_iters_total=0, is_train=False, mean_and_std=mean_and_std, limb_length=limb_length, master=master, experiment_dir=experiment_dir, writer=writer, whole_val_dataloader=whole_val_dataloader, dist_size=dist_size)

    print("Done.")

if __name__ == '__main__':
    args = parse_args()
    print("args: {}".format(args))
    main(args)
