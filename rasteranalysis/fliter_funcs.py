'''
Author: zhang wishing
Date: 2020-08-31 12:56:21
LastEditTime: 2020-09-12 15:00:12
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath:
'''
import numpy as np
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
import findpeaks.utils.stats as stats
import findpeaks


def lee_filter(img, size, mode='reflect'):
    """
    lee滤波算法

    Reference:
    Lee, J. S.: Speckle suppression and analysis for SAR images, Opt. Eng., 25, 636–643, 1986.
    @param img: 输入的影像 numpy数组
    @param size: 滤波窗口大小
    @param mode: 滤波边缘处理方式 详见：https://docs.scipy.org/doc/scipy-0.19.1/reference/generated/scipy.ndimage.uniform_filter.html
    @return 返回滤波后的影像
    """

    img_mean = uniform_filter(img, (size, size), mode=mode)
    img_sqr_mean = uniform_filter(img ** 2, (size, size))

    img_variance = img_sqr_mean - img_mean ** 2
    overall_variance = variance(img)

    np.seterr(divide='ignore', invalid='ignore')

    img_weights = img_variance ** 2 / \
                  (img_variance ** 2 + overall_variance ** 2)
    img_output = img_mean + img_weights * (img - img_mean)

    return img_output.astype(np.int)


def frost_filter(img, damping_factor=2.0, win_size=3):
    '''
    frost 滤波算法
    '''
    findpeaks.frost_filter(img, damping_factor, win_size)


def lee_enhanced_filter(img):
    '''
    lee 增强滤波
    '''
    findpeaks.lee_enhanced_filter(img, win_size=3)


def gamma_map_filter():
    pass
