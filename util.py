import os
import torch
import torch.nn as nn
from torch import float32
from model import *

# Training parameters
useSave = False
version = 1
lr = 1e-4
batch_size = 4
epochs = 15
L1weight = 100
direction = 0       # 0: left -> right로 학습, 1: right -> left로 학습

# Directories
def makeDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

dir_data = 'E:\dataset\pix2pix\edges2shoes'
dir_train = os.path.join(dir_data, 'train')
dir_val = os.path.join(dir_data, 'val')

ckpt_dir = 'C:/Users/msKim/Desktop/pix2pix/ckpt'
log_dir = 'C:/Users/msKim/Desktop/pix2pix/log'
result_dir = 'C:/Users/msKim/Desktop/pix2pix/result'
result_dir_train = os.path.join(result_dir, 'train')
result_dir_val = os.path.join(result_dir, 'val')

# Helper Functions
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

# segmentation data를 output image의 크기에 맞게 crop
def cropimg(seg, output):
    s1, s2 = seg.size()[-2:]
    o1, o2 = output.size()[-2:]
    i1, i2 = (s1-o1)//2, (s2-o2)//2
    segcrop = seg[:, :, i1 : (i1+o1), i2 : (i2+o2)]
    return segcrop

# 네트워크 저장
def save(ckpt_dir, netG, netD, optimG, optimD, epoch, ver):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    torch.save({'netG': netG.state_dict(), 'netD': netD.state_dict(),
                'optimG': optimG.state_dict(), 'optimD': optimD.state_dict()},
                "%s/ver%d_model_epoch%d.pth" % (ckpt_dir, ver, epoch))

# 네트워크 불러오기
def load(ckpt_dir, netG, netD, optimG, optimD):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return netG, netD, optimG, optimD, epoch
    
    ckpt_lst = os.listdir(ckpt_dir)
    #ckpt_lst.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))
    print("USED MODEL :  %s" % ckpt_lst[-1])
    netG.load_state_dict(dict_model['netG'])
    netD.load_state_dict(dict_model['netD'])
    optimG.load_state_dict(dict_model['optimG'])
    optimD.load_state_dict(dict_model['optimD'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return netG, netD, optimG, optimD, epoch

# output pixels의 값을 0또는 1으로 변환
def makePredict(output):
    result = torch.zeros_like(output)
    result[output > 0.4] = 1
    return result

def F1_score(output, seg):
    tp = (seg * output).sum().to(torch.float32)
    tn = ((1 - seg) * (1 - output)).sum().to(torch.float32)
    fp = ((1 - seg) * output).sum().to(torch.float32)
    fn = (seg * (1 - output)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return float(f1)

def Dice_score(output, seg, mode='kid'):
    seg0 = torch.zeros_like(seg)
    seg1 = torch.zeros_like(seg)
    seg2 = torch.zeros_like(seg)
    seg0[seg == 0] = 1
    seg1[seg == 1] = 1
    seg2[seg == 2] = 1
    score0 = F1_score(output[:, 0, :, :], seg0)
    score1 = F1_score(output[:, 1, :, :], seg1)
    if (mode == 'all'):
        score2 = F1_score(output[:, 2, :, :], seg2)
        score = (score0 + score1 + score2) / 3.0
    else :
        score = (score0 + score1) / 2.0

    return float(score)