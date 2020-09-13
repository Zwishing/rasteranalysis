'''
Author: zhang wishing
Date: 2020-08-31 12:47:32
LastEditTime: 2020-09-12 14:43:26
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: 
'''
import numpy as np
from scipy.ndimage import gaussian_filter


def two_peaks_threshold(image, smooth_hist=True, sigma=5, nodata=0):
    """Runs the two peaks threshold algorithm. It selects two peaks
    from the histogram and return the index of the minimum value
    between them.
    双峰算法阈值
    The first peak is deemed to be the maximum value fo the histogram,
    while the algorithm will look for the second peak by multiplying the
    histogram values by the square of the distance from the first peak.
    This gives preference to peaks that are not close to the maximum.

    Reference:
    Parker, J. R. (2010). Algorithms for image processing and
    computer vision. John Wiley & Sons.

    @param image: The input image
    @type image: ndarray
    @param smooth_hist: Indicates whether to smooth the input image
        histogram before finding peaks.
    @type smooth_hist: bool
    @param sigma: The sigma value for the gaussian function used to
        smooth the histogram.
    @type sigma: int

    @return: The threshold between the two founded peaks with the
        minimum histogram value
    @rtype: int
    """
    hist = np.histogram(image, bins=range(256))[0].astype(np.float)

    if smooth_hist:
        hist = gaussian_filter(hist, sigma=sigma)

    f_peak = np.argmax(hist)

    # finding second peak
    s_peak = np.argmax((np.arange(len(hist)) - f_peak) ** 2 * hist)

    thr = np.argmin(hist[min(f_peak, s_peak): max(f_peak, s_peak)])
    thr += min(f_peak, s_peak)

    return thr


def min_err_threshold(image, nodata=0):
    """Runs the minimum error thresholding algorithm.
    最小误差算法阈值
    Reference:
    Kittler, J. and J. Illingworth. ‘‘On Threshold Selection Using Clustering
    Criteria,’’ IEEE Transactions on Systems, Man, and Cybernetics 15, no. 5
    (1985): 652–655.

    @param image: The input image
    @type image: ndarray

    @return: The threshold that minimize the error
    @rtype: int
    """
    # Input image histogram
    hist = np.histogram(image, bins=range(256))[0].astype(np.float)

    # The number of background pixels for each threshold
    w_backg = hist.cumsum()
    w_backg[w_backg == 0] = 1  # to avoid divisions by zero

    # The number of foreground pixels for each threshold
    w_foreg = w_backg[-1] - w_backg
    w_foreg[w_foreg == 0] = 1  # to avoid divisions by zero

    # Cumulative distribution function
    cdf = np.cumsum(hist * np.arange(len(hist)))

    # Means (Last term is to avoid divisions by zero)
    b_mean = cdf / w_backg
    f_mean = (cdf[-1] - cdf) / w_foreg

    # Standard deviations
    b_std = ((np.arange(len(hist)) - b_mean) ** 2 * hist).cumsum() / w_backg
    f_std = ((np.arange(len(hist)) - f_mean) ** 2 * hist).cumsum()
    f_std = (f_std[-1] - f_std) / w_foreg

    # To avoid log of 0 invalid calculations
    b_std[b_std == 0] = 1
    f_std[f_std == 0] = 1

    # Estimating error
    error_a = w_backg * np.log(b_std) + w_foreg * np.log(f_std)
    error_b = w_backg * np.log(w_backg) + w_foreg * np.log(w_foreg)
    error = 1 + 2 * error_a - 2 * error_b

    return np.argmin(error)


def otsu_threshold(image=None, hist=None):
    """ Runs the Otsu threshold algorithm.
        大津算法阈值
    Reference:
    Otsu, Nobuyuki. "A threshold selection method from gray-level
    histograms." IEEE transactions on systems, man, and cybernetics
    9.1 (1979): 62-66.

    @param image: The input image
    @type image: ndarray
    @param hist: The input image histogram
    @type hist: ndarray

    @return: The Otsu threshold
    @rtype int
    """
    if image is None and hist is None:
        raise ValueError('You must pass as a parameter either'
                         'the input image or its histogram')
    # Calculating histogram
    if not hist:
        hist = np.histogram(image, bins=range(256))[0].astype(np.float)
    cdf_backg = np.cumsum(np.arange(len(hist)) * hist)
    w_backg = np.cumsum(hist)  # The number of background pixels
    w_backg[w_backg == 0] = 1  # To avoid divisions by zero
    m_backg = cdf_backg / w_backg  # The means

    cdf_foreg = cdf_backg[-1] - cdf_backg
    w_foreg = w_backg[-1] - w_backg  # The number of foreground pixels
    w_foreg[w_foreg == 0] = 1  # To avoid divisions by zero
    m_foreg = cdf_foreg / w_foreg  # The means

    var_between_classes = w_backg * w_foreg * (m_backg - m_foreg) ** 2

    return np.argmax(var_between_classes)
