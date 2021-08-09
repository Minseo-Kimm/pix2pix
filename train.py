import os
import numpy as np
import torch, gc
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from util import *
from dataloader import *

gc.collect()
torch.cuda.empty_cache()

useSave = False         # 저장된 모델 사용하여 학습 시작

torch.manual_seed(300)

transform_train = transforms.Compose([Resize(shape=(286, 286, 1)), 
                                      RandomCrop((256, 256)),
                                      Normalization(mean=0.5, std=0.5)])
dataset_train = pix2pix_Dataset(dir_train, transform=transform_train, direction=direction)
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
num_data_train = len(dataset_train)
num_batch_train = np.ceil(num_data_train / batch_size)

transform_val = transforms.Compose([Resize(shape=(286, 286, 1)), 
                                      RandomCrop((256, 256)),
                                      Normalization(mean=0.5, std=0.5)])
dataset_val = pix2pix_Dataset(dir_val, transform=transform_val, direction=direction)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, drop_last=True)
num_data_val = len(dataset_val)
num_batch_val = np.ceil(num_data_val / batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G = Generator().to(device)
D = Discriminator().to(device)
fn_loss = nn.CrossEntropyLoss().to(device)
optim_G = torch.optim.Adam(G.parameters(), lr=lr)
optim_D = torch.optim.Adam(D.parameters(), lr=lr)

# 학습 전 저장된 네트워크가 있다면 불러오기
st_epoch = 0
pre_loss = 1
pre_score = 0
if (useSave):
    net, optim, st_epoch, pre_loss = load(ckpt_dir = ckpt_dir, net=net, optim=optim, pre_loss=pre_loss)

# Training
print("TRAINING STARTS")
for epoch in range(st_epoch, epochs):
    net.to(device)
    net.train()
    loss_arr = []

    for batch, data in enumerate(loader_train, 1):
        # forward pass
        seg = data['seg'].to(device=device, dtype=torch.float)
        vol = data['vol'].to(device=device, dtype=torch.float)

        output = net(vol)

        # backward pass
        optim.zero_grad()
        seg = seg.long().squeeze()
        loss = fn_loss(output, seg) # output shape: [3, 3, 512, 512], seg shape: [3, 512, 512]
        loss.backward()
        optim.step()

        # loss 계산
        loss_arr += [loss.item()]
        if (batch % 100 == 0) :
            num = (batch //100) % 10
            print(num, end='')
        if (batch % 500 == 1) :
            res = "TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | Loss %.4f\n" % (epoch + 1, epochs, batch, num_batch_train, np.mean(loss_arr))
            memo.write(res)

    with torch.no_grad():
        print("\nVALIDATION STARTS")
        net.eval()
        loss_arr = []
        acc_arr = []

        for batch, data in enumerate(loader_val, 1):
            # forward pass
            seg = data['seg'].to(device, dtype=torch.float)
            vol = data['vol'].to(device, dtype=torch.float)

            output = net(vol)

            # loss 계산
            loss = fn_loss(output, seg.long().squeeze()).detach().item()
            acc = Dice_score(makePredict(output), seg, mode)
            loss_arr += [loss]
            acc_arr += [acc]
            #print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
            #      ((epoch + 1), epochs, batch, num_batch_val, loss))
    
    loss_mean = np.mean(loss_arr)
    acc_mean = np.mean(acc_arr)
    print("-------------------------------------------------------------")
    res = "EPOCH : %d | MEAN VALID LOSS : %.4f | SCORE : %.4f\n" % ((epoch + 1), loss_mean, acc_mean)
    memo.write(res)
    print("EPOCH : %d | MEAN VALID LOSS : %.4f | SCORE : %.4f" % ((epoch + 1), loss_mean, acc_mean))
    print("-------------------------------------------------------------\n")

    # score가 최소이면 해당 네트워크를 저장
    # from ver9: 모든 네트워크 저장
    if (True):
        save(ckpt_dir=ckpt_dir, net=net.cpu(), optim=optim, epoch=epoch + 1, ver=version, loss=loss_mean)
        pre_score = acc_mean

memo.close()