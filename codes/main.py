import os
import sys
import time
import datetime
import argparse
import shutil
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
# from apex import amp

from DFTR.models import build_model
from DFTR.datasets import SODData
from DFTR import utils
from DFTR import get_loss, get_optimizer, get_scheduler


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def update_lr(epoch, allEpoch, lr, k=1):
    cycle = allEpoch//k or allEpoch
    epoch %=cycle
    return (1 - abs((epoch + 1) / (cycle + 1) * 2 - 1)) * lr


def train(opts, net, LOSS, savepath, begin_epoch=0, log_file=sys.stdout):
    data_opts = opts['dataset']
    img_size = data_opts['img_size']
    val_test = data_opts['val_test']
    datapath = data_opts['prefix']
    data_names = data_opts['train_name']
    train_data = SODData(datapath, data_names, 'train',  img_size, val_test)
    train_loader = DataLoader(train_data, collate_fn=train_data.collate, batch_size=opts.batch_size, shuffle=True)
    val_names = [i.replace('train', 'test') for i in data_names]
    val_data = SODData(datapath, val_names, 'validate',  img_size, val_test)
    val_loader = DataLoader(val_data, collate_fn=val_data.collate, batch_size=opts.batch_size, shuffle=True)

    # network
    net.train(True)
    net.cuda()

    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            print(name)
        elif 'bkbone' in name or 'encoder' in name:
            base.append(param)
        else:
            head.append(param)
    print(f'base: {len(base)},  head: {len(head)}', file=log_file)
    learn_opts = opts.learning
    optim = learn_opts.optimizer
    optim_opts = learn_opts[optim]
    lr = optim_opts['lr']
    params = [{'params':head}]
    if opts.learning.freeze_base:
        for p in net.encoder.parameters():
            p.requires_grad=False
    else:
        params.append({'params':base})
    optimizer = get_optimizer(optim)(params, **optim_opts)
    # net, optimizer = amp.initialize(net, optimizer, opt_level='O1')
    scheduler = None
    if learn_opts.scheduler and learn_opts.scheduler in learn_opts:
        scheduler = get_scheduler(learn_opts.scheduler)(optimizer, **learn_opts[learn_opts.scheduler])
    print('scheduler',scheduler, file=log_file)

    checkpoint_savedir = os.path.join(savepath, 'checkpoints')
    utils.mkdir(checkpoint_savedir)

    best_checkpoint_path = ''

    best_val_loss = float('inf')
    best_train_loss = float('inf')
    train_loss = val_loss = float('inf')

    data_record = {'epoch':[], 'lr':[],'train':[],'val':[]}
    for epoch in range(begin_epoch, opts.epoch):
        # update lr
        if not scheduler:
            optimizer.param_groups[0]['lr'] = update_lr(epoch, opts.epoch, lr, k=2)
            if not opts.learning.freeze_base:
                optimizer.param_groups[1]['lr'] = update_lr(epoch, opts.epoch, opts.learning.base_lr_ratio*lr, k=2)


        train_loss_lst = []
        for step, (image, mask, depth) in enumerate(train_loader):
            image, mask, depth = image.float().cuda(), mask.float().cuda(), depth.float().cuda()

            optimizer.zero_grad()

            outputs = net(image)
            loss, loss1b, loss1u, loss2d, loss1h = LOSS(mask, depth, *outputs)

            if opts.learning.flood>0:
                loss = (loss - opts.learning.flood).abs() + opts.learning.flood

            # with amp.scale_loss(loss, optimizer) as scale_loss:
            #     scale_loss.backward()
            train_loss_lst.append(loss.item())
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            # if step % 30 == 0:
            #     info = '%s | epoch:%d/%d | batch:%04d/%d | lr=%.4f | cur_tr=%.3f | bst_val=%.3f | loss=%.3f | bce=%.3f | region=%.3f | logMSE=%.3f | dec=%.3f ' % (str(datetime.datetime.now()).split('.')[0], epoch+1, opts.epoch, step+1, len(train_loader), optimizer.param_groups[0]['lr'],loss.item(), best_val_loss, loss.item(), loss1b.item(), loss1u.item(), loss2d.item(), loss1h.item())
            #     print(info, file=sys.stdout)
            # break
            # TODO
        train_loss = sum(train_loss_lst)/len(train_loss_lst)
        best_train_loss = min(train_loss, best_train_loss)
        if epoch % opts.show_freq == 0:
            info = '%s | epoch:%d/%d | batch:%04d/%d | lr=%.4f | bst_tr=%.3f | bst_val=%.3f | loss=%.3f | bce=%.3f | region=%.3f | logMSE=%.3f | dec=%.3f ' % (str(datetime.datetime.now()).split('.')[0], epoch+1, opts.epoch, step+1, len(train_loader), optimizer.param_groups[0]['lr'],best_train_loss, best_val_loss, loss.item(), loss1b.item(), loss1u.item(), loss2d.item(), loss1h.item())
            print(info, file=log_file)
            if log_file!=sys.stdout:
                print(info, file=sys.stdout)
        # validate
        if epoch==begin_epoch or (epoch+1) % opts.val_freq == 0:
            net.eval()
            # validate
            val_loss_lst = []
            with torch.no_grad():
                for image, mask, depth in val_loader:
                    image, mask, depth = image.float().cuda(), mask.float().cuda(), depth.float().cuda()
                    outputs = net(image)
                    val_loss = LOSS(mask, depth, *outputs)[0].item()
                    val_loss_lst.append(val_loss)
                    # break
            net.train()
            val_loss = sum(val_loss_lst)/len(val_loss_lst)
            if  val_loss<best_val_loss:
                best_val_loss = val_loss
                name = f'{opts.run_name}_epoch{epoch+1:03d}_train{train_loss:.3f}_val{val_loss:.3f}.pth'
                best_checkpoint_path = os.path.join(checkpoint_savedir, name)
                torch.save(net.state_dict(), best_checkpoint_path)
        if epoch%opts.save_freq==0 or epoch == opts.epoch-1:
            name = f'-{opts.run_name}_epoch{epoch+1:03d}_train{train_loss:.3f}_best{best_val_loss:.3f}.pth'
            torch.save(net.state_dict(), os.path.join(checkpoint_savedir, name))


        data_record['epoch'].append(epoch)
        data_record['lr'] .append(optimizer.param_groups[0]['lr'])
        data_record['train'] .append(train_loss)
        data_record['val'].append(val_loss)
        utils.toYaml(f"{savepath}/train_loss.yaml", data_record)

    return best_checkpoint_path


def test(MODEL, dest, img_size, pre, data_names):
    length = max([len(i) for i in data_names])
    for dataname in data_names:
        test_data = SODData(pre, [dataname], 'test', img_size)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

        ## network
        net = MODEL.cuda()
        net.eval()
        net = nn.DataParallel(net)

        cur_dest = os.path.join(dest, dataname)
        utils.mkdir(cur_dest)
        with torch.no_grad():
            pbar = tqdm(test_loader, ncols=100)
            data_num  = len(pbar)
            for i,(image, mask, shape, name) in enumerate(pbar):
                pbar.set_description(f'{dataname.ljust(length)} {i+1:04d}/{data_num:03d}: {name[0]}')

                image = image.cuda().float()

                torch.cuda.synchronize()
                outputs= net(image, shape)
                mask = outputs[0]
                depth = outputs[4] if len(outputs)>5 else outputs[5]
                torch.cuda.synchronize()
                mask = (torch.sigmoid(mask[0, 0]) * 255).cpu().numpy()
                depth = (torch.sigmoid(depth[0, 0]) * 255).cpu().numpy()
                cv2.imwrite(os.path.join(cur_dest, name[0] + '_mask.png'), np.round(mask))
                cv2.imwrite(os.path.join(cur_dest, name[0] + '_depth.png'), np.round(depth))
                # break


def get_args():
    parser = argparse.ArgumentParser()
    # required
    parser.add_argument("-d", "--run_dir", type=str, required=True)
    parser.add_argument("-r", "--run_name", type=str, required=True)
    parser.add_argument("-p", "--phase", type=str, choices=['train', 'test', 'eval'], required=True)

    # optional
    parser.add_argument("-m", "--model_name", type=str, default='DFTR')
    parser.add_argument("-C", "--config_file", type=str)
    parser.add_argument("-c", "--checkpoint_path", type=str, default='')
    parser.add_argument("-g", "--gpu", type=str, default='0')

    parser.add_argument("-e", "--epoch", type=int, default=200)
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument("-s", "--img_size", type=int, default=352)
    parser.add_argument("-f", "--fusion_depth", type=int, default=1)
    parser.add_argument("--swin_type", choices=['small', 'base', 'base_2', 'large'], default='base')
    parser.add_argument("--down_scale", type=int, default=8)
    parser.add_argument("--residual", action='store_true')
    parser.add_argument("--deep_supervision", action='store_true')
    parser.add_argument("--optimizer", default='sgd')
    parser.add_argument("--scheduler", default='None')
    parser.add_argument("--base_lr_ratio", type=float, default=0.1)
    parser.add_argument("--bce_weight", type=float, default=1)
    parser.add_argument("--flood", type=float, default=0)
    parser.add_argument('-l', "--lr", type=float,  default=0.02)
    return parser.parse_args()


if __name__ == "__main__":
    setup_seed(42)

    args = get_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    run_dir = args.run_dir
    savepath = os.path.join(run_dir, args.run_name)
    utils.mkdir(savepath)

    ''' config '''
    config_file = args.config_file
    if not config_file:
        config_path = os.path.join(savepath, 'config_train.yaml')
        if args.phase == 'test' and os.path.exists(config_path):
            config_file = config_path
        else:
            config_file  = './config.yaml'

    opts = utils.get_config(config_file)

    ''' update config with args '''
    utils.update_config(opts, args)
    if args.phase == 'train':
        utils.toYaml(f"{savepath}/config_train.yaml", opts)
        dest_config = os.path.join(savepath, os.path.basename(config_file))
        if config_file!=dest_config:
            shutil.copy(config_file, dest_config)

    ''' get model params'''
    model_name = opts.model_name.upper()
    if model_name.startswith('DFTR'):
        net_params = opts['DFTR']
        for k,v in opts[f'swin_{opts.swin_type}'].items():
            net_params[k] = v
    else:
        net_params = opts[model_name]
    print('swin type, depths:', opts.swin_type, net_params['depths'])
    print('swin checkpoint:', net_params['base_path'])

    best_path = opts.checkpoint_path

    ''' train '''
    if args.phase=='train':
        begin_epoch=0
        if os.path.exists(opts.checkpoint_path):
            name = os.path.basename(opts.checkpoint_path)
            k = name.find('epoch')
            begin_epoch = int(name[k+5:k+8])
    
        loss = None
        if model_name=='dasnet':
            loss = get_loss('sod_loss')
        else:
            loss = get_loss('sodloss', **opts.learning.sodloss)
            # get_loss('sod_loss2')
        net_params['checkpoint_path'] = best_path
        with open(f'{savepath}/train_{str(datetime.datetime.now()).split(".")[0]}.log','w') as log_file:
            print(opts, file=log_file)
            best_path = train(opts, build_model(model_name,net_params), loss, savepath, begin_epoch, log_file)

    if not os.path.exists(best_path):
        check_dir = os.path.join(savepath, 'checkpoints')
        try:
            checks = [i for i in os.listdir(check_dir) if (i.endswith('.pt') or i.endswith('.pth')) and 'val' in  i]
            checks.sort(key=lambda s: float(s[s.rfind('val')+3:s.rfind('.')]))
            best_path = os.path.join(check_dir, checks[0])
        except Exception as e:
            print(e, check_dir)
            raise Exception('No checkpoint, train first')
    test_path = os.path.join(savepath, 'test_'+os.path.basename(best_path))
    if os.path.exists(test_path) and len(os.listdir(test_path))>1:
        test_path = test_path+'_'+str(datetime.datetime.now())

    ''' test '''
    data_opts = opts['dataset']
    if args.phase!='eval':
        net_params['checkpoint_path'] = best_path
        test(build_model(model_name,net_params), test_path, data_opts['img_size'], data_opts['prefix'], data_opts['test_names'])


    ''' eval '''
    from evaluate import *
    output = f'{savepath}/eval_test_{os.path.basename(best_path)}'
    evaluate_main(test_path, data_opts['prefix'], eval_depth=False, save_img=True, output=output)
