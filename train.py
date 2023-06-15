import argparse
import os

import yaml
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import make_grid, save_image
import datasets
import models
import utils
from statistics import mean
import torch
import torch.distributed as dist

from torch.profiler import profile, record_function, ProfilerActivity

torch.distributed.init_process_group(backend='nccl')
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    if local_rank == 0:
        log('{} dataset: size={}'.format(tag, len(dataset)))
        #for k, v in dataset[0].items():
        #    if k == "image_name":
        #        print(v)
        #    else:
        #        log('  {}: shape={}'.format(k, tuple(v.shape)))
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=False, num_workers=8, pin_memory=True, sampler=sampler)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    #val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    #return train_loader, val_loader
    return train_loader 

def eval_psnr(loader, model, eval_type=None):
    model.train()

    if eval_type == 'f1':
        metric_fn = utils.calc_f1
        metric1, metric2, metric3, metric4 = 'f1', 'auc', 'none', 'none'
    elif eval_type == 'fmeasure':
        metric_fn = utils.calc_fmeasure
        metric1, metric2, metric3, metric4 = 'f_mea', 'mae', 'none', 'none'
    elif eval_type == 'ber':
        metric_fn = utils.calc_ber
        metric1, metric2, metric3, metric4 = 'shadow', 'non_shadow', 'ber', 'none'
    elif eval_type == 'cod':
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'

    if local_rank == 0:
        pbar = tqdm(total=len(loader), leave=False, desc='val')
    else:
        pbar = None

    pred_list = []
    gt_list = []
    for batch in loader:
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = batch['inp']

        pred = torch.sigmoid(model.infer(inp))

        batch_pred = [torch.zeros_like(pred) for _ in range(dist.get_world_size())]
        batch_gt = [torch.zeros_like(batch['gt']) for _ in range(dist.get_world_size())]

        dist.all_gather(batch_pred, pred)
        pred_list.extend(batch_pred)
        dist.all_gather(batch_gt, batch['gt'])
        gt_list.extend(batch_gt)
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    pred_list = torch.cat(pred_list, 1)
    gt_list = torch.cat(gt_list, 1)
    result1, result2, result3, result4 = metric_fn(pred_list, gt_list)

    return result1, result2, result3, result4, metric1, metric2, metric3, metric4

def prepare_training():
    if config.get('resume') is not None:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = config.get('resume') + 1
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
    max_epoch = config.get('epoch_max')
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
    if local_rank == 0:
        log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler

def train(train_loader, model, epoch):
    model.eval()

    if local_rank == 0:
        pbar = tqdm(total=len(train_loader), leave=False, desc='train')
    else:
        pbar = None

    loss_list = []
    for batch in train_loader:
        #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
        #    with record_function("Train"): 
        for k, v in batch.items():
            if k != "image_name":
                batch[k] = v.to(device)
        inp = batch['inp']
        gt = batch['gt']
        image_name = batch['image_name'][0]
        model.set_input(inp, gt)
        model.optimize_parameters()
        #print(batch["image_name"])
        #pred_mask = torch.float32(model.pred_mask > 0.0)
        batch_loss = [torch.zeros_like(model.loss_G) for _ in range(dist.get_world_size())] 
        dist.all_gather(batch_loss, model.loss_G) 
        loss_list.extend(batch_loss) 
        if epoch % 20 == 0: 
            pred_mask = model.pred_mask
            pred_mask[pred_mask==1] = 255.0
            pred_mask[pred_mask==0] == 0.0 
            gt[gt==1] = 255.0
            gt[gt==0] = 0.0
            final_image = torch.cat([pred_mask[0][0].detach().cpu(), gt[0][0].detach().cpu()], dim=1)
            grid_image = make_grid(final_image, ncols=2)
            save_image(grid_image, f"./save/_cod-sam-vit-b/predicted_client_images/{image_name}")
        #print("shape", pred_mask.shape)
        #print("unique mask",torch.unique(pred_mask))
        #pred_mask_np = np.float32(pred_mask[0].detach().cpu().permute(2,1,0).numpy())
        #pred_mask_np[pred_mask_np==1] = 255.0
        #pred_mask_np[pred_mask_np==0] = 0.0 
        #if not cv2.imwrite(f"./save/_cod-sam-vit-b/predicted_images/{image_name}", pred_mask_np):
        #    raise Exception("Image not saved")
        #import ipdb; ipdb.set_trace()
        if pbar is not None:
            pbar.update(1)
       
        #print("************************Cuda time total**********************") 
        #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=16))
        #print("**********************Cpu memory usage*************************")
        #print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=16)) 
    if pbar is not None:
        pbar.close()

    loss = [i.item() for i in loss_list]
    return mean(loss)


def main(config_, save_path, args):
    global config, log, writer, log_info
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    #train_loader, val_loader = make_data_loaders()
    train_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
        
    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    model.optimizer = optimizer
    lr_scheduler = CosineAnnealingLR(model.optimizer, config['epoch_max'], eta_min=config.get('lr_min'))

    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model,
    device_ids=[args.local_rank],
    output_device=args.local_rank,
    find_unused_parameters=True,
    broadcast_buffers=False
    )
    model = model.module
    sam_checkpoint = torch.load(config['sam_checkpoint'])
    model.load_state_dict(sam_checkpoint, strict=False)
    for name, para in model.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)
    size_model = 0
    for param in model.parameters():
        if param.data.is_floating_point():
            size_model += param.numel() * torch.finfo(param.data.dtype).bits
        else:
            size_model += param.numel() * torch.iinfo(param.data.dtype).bits
    print(f"model size: {size_model} / bit | {size_model / 8e6:.2f} / MB")
    
    if local_rank == 0:
        model_total_params = sum(p.numel() for p in model.parameters())
        model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    max_val_v = -1e18 if config['eval_type'] != 'ber' else 1e8
    timer = utils.Timer()
    
    for epoch in range(epoch_start, epoch_max + 1):
         
        train_loader.sampler.set_epoch(epoch)
        t_epoch_start = timer.t()
        train_loss_G = train(train_loader, model, epoch)
        lr_scheduler.step()
        if local_rank == 0:
            log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            log_info.append('train: loss={:.4f}'.format(train_loss_G))
            writer.add_scalars('loss', {'hiertext_train_loss_line': train_loss_G}, epoch)
            print(f"Epoch: {epoch}, Loss: {train_loss_G}")
            model_spec = config['model']
            model_spec['sd'] = model.state_dict()
            optimizer_spec = config['optimizer']
            optimizer_spec['sd'] = optimizer.state_dict()

            save(config, model, save_path, 'last')
#        if (epoch_val is not None) and (epoch % epoch_val == 0):
#            result1, result2, result3, result4, metric1, metric2, metric3, metric4 = eval_psnr(val_loader, model,
#                eval_type=config.get('eval_type'))
#
#            if local_rank == 0:
#                log_info.append('val: {}={:.4f}'.format(metric1, result1))
#                writer.add_scalars(metric1, {'val': result1}, epoch)
#                log_info.append('val: {}={:.4f}'.format(metric2, result2))
#                writer.add_scalars(metric2, {'val': result2}, epoch)
#                log_info.append('val: {}={:.4f}'.format(metric3, result3))
#                writer.add_scalars(metric3, {'val': result3}, epoch)
#                log_info.append('val: {}={:.4f}'.format(metric4, result4))
#                writer.add_scalars(metric4, {'val': result4}, epoch)
#
#                if config['eval_type'] != 'ber':
#                    if result1 > max_val_v:
#                        max_val_v = result1
#                        save(config, model, save_path, 'best')
#                else:
#                    if result3 < max_val_v:
#                        max_val_v = result3
#                        save(config, model, save_path, 'best')
#
#                t = timer.t()
#                prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
#                t_epoch = utils.time_text(t - t_epoch_start)
#                t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
#                log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))
#
#                log(', '.join(log_info))
#                writer.flush()


def save(config, model, save_path, name):
    if config['model']['name'] == 'segformer' or config['model']['name'] == 'setr':
        if config['model']['args']['encoder_mode']['name'] == 'evp':
            prompt_generator = model.encoder.backbone.prompt_generator.state_dict()
            decode_head = model.encoder.decode_head.state_dict()
            torch.save({"prompt": prompt_generator, "decode_head": decode_head},
                       os.path.join(save_path, f"prompt_epoch_{name}.pth"))
        else:
            torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/train/setr/train_setr_evp_cod.yaml")
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        if local_rank == 0:
            print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path, args=args)
