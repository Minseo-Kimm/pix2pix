import os
import numpy as np
import torch, gc
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from util import *
from dataloader import *

writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

gc.collect()
torch.cuda.empty_cache()
torch.manual_seed(300)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create Dataset & Network
"""
transform_train = transforms.Compose([Resize(shape=(286, 286, 1)), 
                                      RandomCrop((256, 256)),
                                      Normalization(mean=0.5, std=0.5), ToTensor()])"""
transform_train_imsi = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])
dataset_train = pix2pix_Dataset(dir_train, transform=transform_train_imsi, direction=direction)
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
num_data_train = len(dataset_train)
num_batch_train = np.ceil(num_data_train / batch_size)

"""
transform_val = transforms.Compose([Resize(shape=(286, 286, 1)), 
                                      RandomCrop((256, 256)),
                                      Normalization(mean=0.5, std=0.5), ToTensor()])"""
transform_val_imsi = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])
dataset_val = pix2pix_Dataset(dir_val, transform=transform_val_imsi, direction=direction)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, drop_last=True)
num_data_val = len(dataset_val)
num_batch_val = np.ceil(num_data_val / batch_size)

G = Generator(in_chs=3, out_chs=3).to(device)
D = Discriminator(in_chs=6, out_chs=1).to(device)

# weight initialization ???

# Loss Functions & Optimizer
fn_L1 = nn.L1Loss().to(device)
fn_GAN = nn.BCELoss().to(device)

optim_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optim_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# 학습 전 저장된 네트워크가 있다면 불러오기
st_epoch = 0
if (useSave):
    G, D, optim_G, optim_D, st_epoch = load(ckpt_dir=ckpt_dir,
                                            netG=G, netD=D, optimG=optim_G, optimD=optim_D)

# Training
print("TRAINING STARTS")
for epoch in range(st_epoch, epochs):
    G.train()
    D.train()
    loss_G_L1_train = []
    loss_G_GAN_train = []
    loss_D_real_train = []
    loss_D_fake_train = []

    for batch, data in enumerate(loader_train, 1):
        # forward pass
        label = data['label'].to(device)
        input = data['input'].to(device)
        output = G(input)

        # backward D
        optim_D.zero_grad()
        real = torch.cat([input, label], dim=1)
        fake = torch.cat([input, output], dim=1)

        pred_real = D(real)
        pred_fake = D(fake.detach())
        loss_D_real = fn_GAN(pred_real, torch.ones_like(pred_real))
        loss_D_fake = fn_GAN(pred_fake, torch.zeros_like(pred_fake))
        loss_D = 0.5 * (loss_D_real + loss_D_fake)

        loss_D.backward()
        optim_D.step()

        # backward G
        optim_G.zero_grad()
        fake = torch.cat([input, output], dim=1)
        
        pred_fake = D(fake)
        loss_G_GAN = fn_GAN(pred_fake, torch.ones_like(pred_fake))
        loss_G_L1 = fn_L1(output, label)
        loss_G = loss_G_GAN + L1weight * loss_G_L1

        loss_G.backward()
        optim_G.step()

        # loss 계산
        loss_G_L1_train += [loss_G_L1.item()]
        loss_G_GAN_train += [loss_G_GAN.item()]
        loss_D_real_train += [loss_D_real.item()]
        loss_D_fake_train += [loss_D_fake.item()]

        if (batch % 10 == 0) :
            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | "
                  "G L1 : %.4f | G GAN : %.4f | "
                  "D REAL : %.4f | D FAKE : %.4f " %
                  (epoch + 1, epochs, batch, num_batch_train,
                   np.mean(loss_G_L1_train), np.mean(loss_G_GAN_train),
                   np.mean(loss_D_real_train), np.mean(loss_D_fake_train)))
        
        if (batch % 20 == 0) :
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5)).squeeze()
            label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5)).squeeze()
            output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5)).squeeze()

            input = np.clip(input, a_min=0, a_max=1)
            label = np.clip(label, a_min=0, a_max=1)
            output = np.clip(output, a_min=0, a_max=1)

            batchnum = num_batch_train * epoch + batch
            res_path = os.path.join(result_dir_train, 'ver_%03d' % version)
            makeDir(res_path)
            plt.imsave(os.path.join(res_path, '%04d_input.png' % batchnum), input[0])
            plt.imsave(os.path.join(res_path, '%04d_label.png' % batchnum), label[0])
            plt.imsave(os.path.join(res_path, '%04d_output.png' % batchnum), output[0])

            writer_train.add_image('input', input, batchnum, dataformats='NHWC')
            writer_train.add_image('label', label, batchnum, dataformats='NHWC')
            writer_train.add_image('output', output, batchnum, dataformats='NHWC')
    
    # epoch 1번 돈 후 loss 기록
    writer_train.add_scalar('loss_G_L1', np.mean(loss_G_L1_train), epoch+1)
    writer_train.add_scalar('loss_G_GAN', np.mean(loss_G_GAN_train), epoch+1)
    writer_train.add_scalar('loss_D_real', np.mean(loss_D_real_train), epoch+1)
    writer_train.add_scalar('loss_D_fake', np.mean(loss_D_fake_train), epoch+1)

    with torch.no_grad():
        print("\nVALIDATION STARTS")
        G.eval()
        D.eval()
        loss_G_L1_val = []
        loss_G_GAN_val = []
        loss_D_real_val = []
        loss_D_fake_val = []

        for batch, data in enumerate(loader_val, 1):
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)
            output = G(input)

            # loss D 계산
            real = torch.cat([input, label], dim=1)
            fake = torch.cat([input, output], dim=1)

            pred_real = D(real)
            pred_fake = D(fake.detach())

            loss_D_real = fn_GAN(pred_real, torch.ones_like(pred_real))
            loss_D_fake = fn_GAN(pred_fake, torch.zeros_like(pred_fake))
            loss_D = 0.5 * (loss_D_real + loss_D_fake)

            # loss G 계산
            fake = torch.cat([input, output], dim=1)
            pred_fake = D(fake)

            loss_G_L1 = fn_L1(output, label)
            loss_G_GAN = fn_GAN(pred_fake, torch.ones_like(pred_fake))
            loss_G = loss_G_GAN + L1weight * loss_G_L1

            # loss 계산
            loss_G_L1_val += [loss_G_L1.item()]
            loss_G_GAN_val += [loss_G_GAN.item()]
            loss_D_real_val += [loss_D_real.item()]
            loss_D_fake_val += [loss_D_fake.item()]

            if (batch % 5 == 0) :
                print("VAL: EPOCH %04d / %04d | BATCH %04d / %04d | "
                    "G L1 : %.4f | G GAN : %.4f | "
                    "D REAL : %.4f | D FAKE : %.4f " %
                    (epoch + 1, epochs, batch, num_batch_val,
                    np.mean(loss_G_L1_val), np.mean(loss_G_GAN_val),
                    np.mean(loss_D_real_val), np.mean(loss_D_fake_val)))
            
            if (batch % 10 == 0) :
                input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5)).squeeze()
                label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5)).squeeze()
                output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5)).squeeze()

                input = np.clip(input, a_min=0, a_max=1)
                label = np.clip(label, a_min=0, a_max=1)
                output = np.clip(output, a_min=0, a_max=1)

                batchnum = num_batch_val * epoch + batch
                res_path = os.path.join(result_dir_val, 'ver_%03d' % version)
                makeDir(res_path)
                plt.imsave(os.path.join(res_path, '%04d_input.png' % batchnum), input[0])
                plt.imsave(os.path.join(res_path, '%04d_label.png' % batchnum), label[0])
                plt.imsave(os.path.join(res_path, '%04d_output.png' % batchnum), output[0])

                writer_train.add_image('input', input, batchnum, dataformats='NHWC')
                writer_train.add_image('label', label, batchnum, dataformats='NHWC')
                writer_train.add_image('output', output, batchnum, dataformats='NHWC')
        
        # epoch 1번 돈 후 loss 기록
        writer_val.add_scalar('loss_G_L1', np.mean(loss_G_L1_val), epoch+1)
        writer_val.add_scalar('loss_G_GAN', np.mean(loss_G_GAN_val), epoch+1)
        writer_val.add_scalar('loss_D_real', np.mean(loss_D_real_val), epoch+1)
        writer_val.add_scalar('loss_D_fake', np.mean(loss_D_fake_val), epoch+1)

    if (epoch % 3 or epoch == epochs - 1):
        save(ckpt_dir=ckpt_dir, netG=G, netD=D, optimG=optim_G, optimD=optim_D, epoch=epoch, ver=version)
    
writer_train.close()
writer_val.close()