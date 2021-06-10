import numpy as np
# import cv2
# from torch.autograd import Variable
# import math
from skimage.metrics import peak_signal_noise_ratio as cal_psnr
from skimage.metrics import structural_similarity as cal_ssim


def cal_metrics(real_y_list, fake_y_list, return_total=False):
    total_psnr = np.zeros(len(real_y_list))
    total_ssim = np.zeros(len(real_y_list))

    for idx in range(len(real_y_list)):
        # total_psnr[idx] = cal_psnr(real_y_list[idx], fake_y_list[idx])
        total_psnr[idx] = cal_psnr(real_y_list[idx], fake_y_list[idx], data_range=1)
        # total_ssim[idx] = cal_ssim(real_y_list[idx], fake_y_list[idx])
        total_ssim[idx] = cal_ssim(real_y_list[idx], fake_y_list[idx], data_range=1)

    if return_total:
        return total_psnr.mean(), total_ssim.mean(), total_psnr, total_ssim
    else:
        return total_psnr.mean(), total_ssim.mean()
