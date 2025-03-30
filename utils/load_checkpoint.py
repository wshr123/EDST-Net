import os
import hashlib
import requests
from jax.example_libraries.optimizers import optimizer
from tqdm import tqdm
import torch.nn as nn

import torch


def load_detr_weights(model, pretrain_dir, cfg):
    checkpoint = torch.load(pretrain_dir, map_location='cpu')
    # print(checkpoint.keys())
    checkpoint = adjust_detr_keys(checkpoint)  # todo
    model_dict = model.state_dict() #返回一个字典，包括模型所有的可学习参数
    # print(checkpoint['model'].keys())
    pretrained_dict = {}
    for k, v in checkpoint['model'].items():    #.ietm：返回键值对
        if k.split('.')[1] == 'transformer':
            pretrained_dict.update({k: v})
        elif k.split('.')[1] == 'bbox_embed':
            pretrained_dict.update({k: v})
        elif k.split('.')[1] == 'query_embed':
            if not cfg.CONFIG.MODEL.SINGLE_FRAME:
                query_size = cfg.CONFIG.MODEL.QUERY_NUM * (cfg.CONFIG.MODEL.TEMP_LEN // cfg.CONFIG.MODEL.DS_RATE)
            else:
                query_size = cfg.CONFIG.MODEL.QUERY_NUM     #10
            # pretrained_dict.update({k: v[:query_size]})
            pretrained_dict.update({k: v[:query_size]})
            # if query_size == model.module.query_embed.weight.shape[0]: continue
            if query_size == model.query_embed.weight.shape[0]: continue
            # if v.shape[0] < model.module.query_embed.weight.shape[0]:  # In case the pretrained model does not align
            if v.shape[0] < model.query_embed.weight.shape[0]:  #320
                # query_embed_zeros = torch.zeros(model.module.query_embed.weight.shape)
                query_embed_zeros = torch.zeros(model.query_embed.weight.shape) #320,256
                pretrained_dict.update({k: query_embed_zeros})
            else:
                # pretrained_dict.update({k: v[:model.module.query_embed.weight.shape[0]]})
                pretrained_dict.update({k: v[:model.query_embed.weight.shape[0]]})

    pretrained_dict_ = {k: v for k, v in pretrained_dict.items() if k in model_dict}    #pretrained_dict为空？
    unused_dict = {k: v for k, v in pretrained_dict.items() if not k in model_dict}
    # not_found_dict = {k: v for k, v in model_dict.items() if not k in pretrained_dict}

    print("detr unused model layers:", unused_dict.keys())
    # print("not found layers:", not_found_dict.keys())

    model_dict.update(pretrained_dict_)     #更新model_dict中对应的值
    model.load_state_dict(model_dict)
    print("load pretrain success")
def load_model(model, cfg, load_fc=True):
    """
    Load pretrained model weights.
    """
    if os.path.isfile(cfg.CONFIG.MODEL.PRETRAINED_PATH):
        print("=> loading checkpoint '{}'".format(cfg.CONFIG.MODEL.PRETRAINED_PATH))
        if cfg.DDP_CONFIG.GPU is None:
            checkpoint = torch.load(cfg.CONFIG.MODEL.PRETRAINED_PATH)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(cfg.DDP_CONFIG.GPU)
            checkpoint = torch.load(cfg.CONFIG.MODEL.PRETRAINED_PATH, map_location=loc)
        model_dict = model.state_dict()
        if not load_fc:
            del model_dict['module.fc.weight']
            del model_dict['module.fc.bias']
        # for k, v in checkpoint['model'].items() :
        #     print(k)
        # for k, v in model_dict.items():
        #     print(k)
        # print(checkpoint.keys())
        checkpoint = adjust_checkpoint_keys(checkpoint) #todo
        pretrained_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict}
        unused_dict = {k: v for k, v in checkpoint['model'].items() if not k in model_dict}
        not_found_dict = {k: v for k, v in model_dict.items() if not k in checkpoint['model']}
        print("unused model layers:", unused_dict.keys())
        print("not found layers:", not_found_dict.keys())
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(cfg.CONFIG.MODEL.PRETRAINED_PATH, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(cfg.CONFIG.MODEL.PRETRAINED_PATH))

    return model, None

 #todo 加载的checpoint层的名字匹配不上，要把module删掉

def adjust_detr_keys(checkpoint):
    d_optimizer = checkpoint['optimizer']
    d_lr_scheduler = checkpoint['lr_scheduler']
    d_epoch = checkpoint['epoch']
    d_args = checkpoint['args']
    new_checkpoint = {
        'model': {},  # 模型部分可以在后面填充
        'lr_scheduler': d_optimizer,
        'max_accuracy': d_lr_scheduler,
        'epoch': d_epoch,
        'config': d_args,
    }
    for k, v in checkpoint['model'].items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_checkpoint['model'][new_key] = v

    return new_checkpoint

def adjust_checkpoint_keys(checkpoint):
    d_optimizer = checkpoint['optimizer']
    d_lr_scheduler = checkpoint['lr_scheduler']
    d_epoch = checkpoint['epoch']
    d_config = checkpoint['config']
    d_max_accuracy = checkpoint['max_accuracy']
    new_checkpoint = {
        'model': {},  # 模型部分可以在后面填充
        'optimizer': d_optimizer,
        'lr_scheduler': d_lr_scheduler,
        'max_accuracy': d_max_accuracy,
        'epoch': d_epoch,
        'config': d_config,
    }
    for k, v in checkpoint['model'].items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_checkpoint['model'][new_key] = v

    return new_checkpoint

def save_checkpoint(cfg, epoch, model, max_accuracy, optimizer, lr_scheduler):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': cfg}

    model_save_dir = os.path.join(cfg.CONFIG.LOG.BASE_PATH,
                                  cfg.CONFIG.LOG.EXP_NAME,
                                  cfg.CONFIG.LOG.SAVE_DIR)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    print('Saving model at epoch %d to %s' % (epoch, model_save_dir))

    save_path = os.path.join(model_save_dir, f'ckpt_epoch_{epoch}.pth')
    torch.save(save_state, save_path)
