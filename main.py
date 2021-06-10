import os
from glob import glob
import sys
import argparse
import numpy as np
import GPUtil
from tqdm import tqdm
from datetime import datetime
import logging

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter

from model import *
# from model_ import *
from data import MRI_7t
from pred import prediction, prediction_self
from metrics import cal_metrics

torch.cuda.empty_cache()
########## GPU Configuration
GPU = -1
if GPU == -1:
    devices = "%d" % GPUtil.getFirstAvailable(order="memory")[0]
else:
    devices = "%d" % GPU
os.environ["CUDA_VISIBLE_DEVICES"] = devices
# os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


########## Argument
parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='train')

parser.add_argument('--path_dataset', type=str, default='/home/ubuntu/jelee/dataset/7T_data_01')
# parser.add_argument('--path_dataset', type=str, default='/DataCommon3/jelee/7T_data_01')

parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--val_num', type=int, default=3)
# parser.add_argument('--re_size', type=int, nargs='+', default=[48, 64, 68])

parser.add_argument('--nf', type=int, default=16)
parser.add_argument('--lr_g', type=float, default=0.0002)
parser.add_argument('--lr_t', type=float, default=0.0002)
parser.add_argument('--lr_d', type=float, default=0.0002)
parser.add_argument('--lambda_1', type=int, default=100) # Weight of voxel-wise loss between fake image and real image
# parser.add_argument('--lambda_2', type=int, default=100)
parser.add_argument('--checkpoint_sample', type=int, default=10)

args = parser.parse_args()



def logger_setting(save_path):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(filename=f'{save_path}/log.log')

    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # stream_handler.setFormatter(formatter)
    # file_handler.setFormatter(formatter)

    stream_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Block') == -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def train():

    ##### Directory
    datetime_train = datetime.today().strftime('%Y%m%d_%H%M%S')
    dir_log = f'./log/{datetime_train}'
    dir_model = f'./log/{datetime_train}/model'
    dir_tboard = f'./log/{datetime_train}/tboard'
    dir_result = f'./log/{datetime_train}/result_valid'

    directory = [dir_log, dir_model, dir_tboard, dir_result]
    for dir in directory:
            os.makedirs(dir, exist_ok=True)


    ##### Training Log
    logger = logger_setting(dir_log)
    logger.debug('============================================')
    logger.debug('Batch Size: %d' % args.batch_size)
    logger.debug('Epoch: %d' % args.epochs)

    writer = SummaryWriter(dir_tboard)


    ##### Initialize
    # cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    loss_L1 = nn.L1Loss()
    loss_MSE = nn.MSELoss()
    loss_BCE = nn.BCELoss()
    sigmoid = nn.Sigmoid()


    ##### Model
    # generator = GeneratorUNet().to(device)
    # discriminator = Discriminator().to(device)
    # if cuda:
    #     generator = generator.to(device)
    #     discriminator = discriminator.to(device)
    generator = nn.DataParallel(GeneratorUNet()).to(device)
    teacher = nn.DataParallel(TeacherUNet()).to(device)
    discriminator = nn.DataParallel(Discriminator()).to(device)

    generator.apply(weights_init_normal)
    teacher.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    optimizer_T = torch.optim.Adam(teacher.parameters(), lr=args.lr_t, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    logger.debug('============================================')
    logger.debug('[Generator]')
    logger.debug(str(generator))
    logger.debug(f'Optimizer: {type(optimizer_G)}')
    logger.debug('Learning Rate: %.5f' % args.lr_g)
    logger.debug('[Teacher]')
    logger.debug(str(teacher))
    logger.debug(f'Optimizer: {type(optimizer_T)}')
    logger.debug('Learning Rate: %.5f' % args.lr_t)
    logger.debug('[Discriminator]')
    logger.debug(str(discriminator))
    logger.debug(f'Optimizer: {type(optimizer_D)}')
    logger.debug('Learning Rate: %.5f' % args.lr_d)
    logger.debug('[Loss]')
    logger.debug('Guide Loss: MSE')
    logger.debug('Adv Loss: MSE')
    logger.debug('Voxel-wise Loss: L1')
    logger.debug('Lambda of Voxel-wise Loss: %d' % args.lambda_1)
    # logger.debug('Lambda of Guide Loss: %d' % args.lambda_2)
    logger.debug('============================================')

    ##### Dataset Load
    # data_path = sorted(glob(f'{args.path_dataset}/*/'))
    # dataset = MRIDataset(args.path_dataset)
    # # train_dataset, val_dataset = random_split(dataset, [14, 1])
    # train_dataset = []
    # val_dataset = []
    # for idx, data in enumerate(dataset):
    #     if idx == 0:
    #         val_dataset.append(data)
    #     else:
    #         train_dataset.append(data)

    all_idx = list(range(1, 16))
    val_idx = np.random.choice(all_idx, args.val_num, replace=False)
    # val_idx = [1]

    train_data_path = []
    val_data_path = []
    for folder_name in sorted(os.listdir(args.path_dataset)):
        _, patient_id = folder_name.split('_')  # folder_name example: S_01
        if int(patient_id) in val_idx:
            val_data_path.append(f'{args.path_dataset}/{folder_name}')
        else:
            train_data_path.append(f'{args.path_dataset}/{folder_name}')

    logger.info(f'''Valid data: {[path.split('/')[-1] for path in val_data_path]}''')
    logger.info(f'''Train data: {[path.split('/')[-1] for path in train_data_path]}''')

    train_dataset = MRI_7t(train_data_path)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = MRI_7t(val_data_path)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    ##### Training
    logger.debug('============================================')

    valid_best = {'epoch': 0, 'psnr': 0, 'ssim': 0}
    generator.train()
    teacher.train()
    discriminator.train()

    for epoch in tqdm(range(1, args.epochs + 1), desc='Epoch'):
        valid_update = False
        for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Batch'):
            logger.debug(f'[Epoch: {epoch} | Batch No.: {i}]')

            real_x = Variable(batch['x']).to(device)
            real_y = Variable(batch['y']).to(device)
            # if cuda:
            #     real_x = real_x.to(device)
            #     real_y = real_y.to(device)

            # ------------------------------
            # Train teacher
            # ------------------------------
            guide_u1, guide_u2, guide_u3, guide_u4, guide_y = teacher(real_y)
            loss_T = loss_L1(real_y, guide_y)

            logger.debug(f'[Teacher]')
            logger.debug(
                f'Loss: {round(loss_T.item(), 4)}'
            )

            optimizer_T.zero_grad()
            loss_T.backward()
            optimizer_T.step()

            del guide_u1, guide_u2, guide_u3, guide_u4, guide_y

            # ------------------------------
            # Train discriminator
            # ------------------------------

            # fake_y = generator(real_x)
            fake_u1, fake_u2, fake_u3, fake_u4, fake_y = generator(real_x)

            # Real loss
            pred_real = discriminator(real_y)
            valid = Variable(Tensor(np.ones(pred_real.size())), requires_grad=False)
            loss_D_real = loss_MSE(pred_real, valid)
            # loss_D_real = loss_L1(pred_real, valid)

            # Fake loss
            pred_fake = discriminator(fake_y)
            fake = Variable(Tensor(np.zeros(pred_fake.size())), requires_grad=False)
            loss_D_fake = loss_MSE(pred_fake, fake)
            # loss_D_fake = loss_L1(pred_fake, fake)

            # Total loss
            loss_D = 0.5 * (loss_D_real + loss_D_fake)

            logger.debug(f'[Discriminator]')
            logger.debug(
                f'Real Loss: {round(loss_D_real.item(), 4)} | Fake Loss: {round(loss_D_fake.item(), 4)} | Total: {round(loss_D.item(), 4)}'
            )

            # # acc 생각해보기. ! fid score ! ncc score (normalized cross correlation) !
            # d_real_acc = torch.ge(pred_real.squeeze(), 0.5).float()
            # d_fake_acc = torch.le(pred_fake.squeeze(), 0.5).float()
            # d_total_acc = torch.mean(torch.cat((d_real_acc, d_fake_acc), 0))
            #
            # if d_total_acc <= args.d_threshold:
            #     optimizer_D.zero_grad()
            #     loss_D.backward()
            #     optimizer_D.step()

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            del fake_u1, fake_u2, fake_u3, fake_u4, fake_y
            del pred_real, pred_fake

            # ------------------------------
            # Train generator
            # ------------------------------

            # loss 업데이트 확인하기 ! D vs G. D가 freezing -> G update / D update <- G freeze 확인

            # optimizer_D.zero_grad()
            # optimizer_G.zero_grad()

            # fake_y = generator(real_x)
            fake_u1, fake_u2, fake_u3, fake_u4, fake_y = generator(real_x)
            guide_u1, guide_u2, guide_u3, guide_u4, guide_y = teacher(real_y)

            # Knowledge Distillation loss
            # loss_G_guide = loss_MSE(fake_u1, guide_u1) + loss_MSE(fake_u2, guide_u2) + loss_MSE(fake_u3, guide_u3) + loss_MSE(fake_u4, guide_u4) + loss_MSE(fake_y, guide_y)
            loss_G_guide = loss_MSE(fake_u1, guide_u1) + loss_MSE(fake_u2, guide_u2) + loss_MSE(fake_u3, guide_u3) + loss_MSE(fake_u4, guide_u4)
            # loss_G_guide = loss_L1(fake_u1, guide_u1) + loss_L1(fake_u2, guide_u2) + loss_L1(fake_u3, guide_u3) + loss_L1(fake_u4, guide_u4) + loss_L1(fake_y, guide_y)

            # GAN loss
            pred_fake = discriminator(fake_y)
            loss_G_fake = loss_MSE(pred_fake, valid)
            # loss_G_fake = loss_L1(pred_fake, valid)
            loss_G_voxel = args.lambda_1 * loss_L1(fake_y, real_y)

            # Total loss
            loss_G = loss_G_fake + loss_G_voxel + loss_G_guide

            logger.debug(f'[Generator]')
            logger.debug(
                f'Guide Loss: {round(loss_G_guide.item(), 4)} | Adv Loss: {round(loss_G_fake.item(), 4)} | Voxel-wise Loss: {round(loss_G_voxel.item(), 4)} | Total: {round(loss_G.item(), 4)}'
            )

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

        if epoch % args.checkpoint_sample == 0:
            real_y_list, fake_y_list = prediction(val_dataloader, generator, device,
                                                  save_pred_path=f'{dir_result}/{epoch}_')
        else:
            real_y_list, fake_y_list = prediction(val_dataloader, generator, device)

        val_psnr, val_ssim = cal_metrics(real_y_list, fake_y_list)
        if valid_best['psnr'] < val_psnr and valid_best['ssim'] < val_ssim:
            torch.save(generator.state_dict(), f'{dir_model}/generator.pth')
            torch.save(teacher.state_dict(), f'{dir_model}/teacher.pth')
            torch.save(discriminator.state_dict(), f'{dir_model}/discriminator.pth')
            valid_best['epoch'] = epoch
            valid_best['psnr'] = val_psnr
            valid_best['ssim'] = val_ssim
            valid_update = True

        real_y_list, self_y_list = prediction_self(val_dataloader, teacher, device)
        val_psnr_self, val_ssim_self = cal_metrics(real_y_list, self_y_list)

        logger.debug('--------------------------------------------')
        logger.info(f'[Epoch: {epoch}/{args.epochs}]')
        logger.info(
            f'D loss: {round(loss_D.item(), 4)} | G loss: {round(loss_G.item(), 4)} | val_psnr: {round(val_psnr, 4)} | val_ssim: {round(val_ssim, 4)} | valid_update: {str(valid_update)}'
            )
        logger.info(
            f'T loss: {round(loss_T.item(), 4)} | val_psnr_self: {round(val_psnr_self, 4)} | val_ssim_self: {round(val_ssim_self, 4)}'
            )
        logger.debug('--------------------------------------------')

        writer.add_scalar('T loss', loss_T.item(), epoch)
        writer.add_scalar('D loss', loss_D.item(), epoch)
        writer.add_scalar('G loss', loss_G.item(), epoch)
        writer.add_scalar('val_psnr', val_psnr, epoch)
        writer.add_scalar('val_ssim', val_ssim, epoch)

        del real_y_list, fake_y_list, self_y_list

    writer.close()

    logger.info('============================================')
    logger.info(f'[Best Performance of Validation]')
    logger.info(f'''Epoch: {valid_best['epoch']} | PSNR: {valid_best['psnr']} | SSIM: {valid_best['ssim']}''')

    del teacher, generator, discriminator

    logger.debug(f'[Prediction for All Data]')
    all_psnr, all_ssim = pred_all(model_path=f'{dir_model}/generator.pth')
    logger.debug(f'PSNR: {all_psnr} | SSIM: {all_ssim}')
    logger.info('============================================')

    torch.cuda.empty_cache()


def pred_all(model_path=False):
    generator = nn.DataParallel(GeneratorUNet()).to(device)

    if model_path:
        generator.load_state_dict(torch.load(model_path))
    else:
        model_path = sorted(glob(f'./log/*/model/generator.pth'))[-1]
        generator.load_state_dict(torch.load(model_path))

    model_dtime = model_path.split('/')[2]

    # cuda = True if torch.cuda.is_available() else False
    # if cuda:
    #     generator = generator.to(device)
    generator = generator.to(device)

    dir_all = f'./result_all/{model_dtime}/'
    os.makedirs(dir_all, exist_ok=True)

    logger_all = logger_setting(dir_all)
    logger_all.info(f'Model: {model_path}')

    all_dataset = MRI_7t(sorted(glob(f'{args.path_dataset}/*')))
    all_dataloader = DataLoader(all_dataset, batch_size=1, shuffle=False)
    real_y_list, fake_y_list, patient_ids = prediction(all_dataloader, generator, device, dir_all)
    psnr, ssim, total_psnr, total_ssim = cal_metrics(real_y_list, fake_y_list, return_total=True)

    np.save(f'{dir_all}total_psnr', total_psnr)
    np.save(f'{dir_all}total_ssim', total_ssim)

    logger_all.info('[Patient ID | PSNR | SSIM]')
    for idx, patient_id in enumerate(patient_ids):
        logger_all.info(f'{patient_id} | {total_psnr[idx]} | {total_ssim[idx]}')

    logger_all.info('[Prediction for All Data]')
    logger_all.info(f'PSNR: {psnr} | SSIM: {ssim}')

    return psnr, ssim


if __name__ == "__main__":
    if args.mode == 'train':
        train()
    elif args.mode == 'pred':
        pred_all()
    else:
        print('[--mode] option: train / pred')

