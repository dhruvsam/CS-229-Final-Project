import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter

def load_dataset(path='/Users/dhruvsamant/Desktop/MLProject/Resnet/train.json'):
     train=pd.read_json(path)
     train_images = train.apply(lambda c_row: [np.stack([c_row['band_1'],c_row['band_2']], -1).reshape((75,75,2))],1)
     train_images = np.stack(train_images).squeeze()
     return train, train_images


def decibel_to_linear(band):
     # convert to linear units
    return np.power(10,np.array(band)/10)

def linear_to_decibel(band):
    return 10*np.log10(band)




# implement the Lee Filter for a band in an image already reshaped into the proper dimensions
def lee_filter(band, window, var_noise = 0.25):
        # band: SAR data to be despeckled (already reshaped into image dimensions)
        # window: descpeckling filter window (tuple)
        # default noise variance = 0.25
        # assumes noise mean = 0

        mean_window = uniform_filter(band, window)
        mean_sqr_window = uniform_filter(band**2, window)
        var_window = mean_sqr_window - mean_window**2


        weights = var_window / (var_window + var_noise)
        band_filtered = mean_window + weights*(band - mean_window)
        return band_filtered

def E_lee_filter(band, window, var_noise = 0.25,damp = 1, numL = 1):
        # band: SAR data to be despeckled (already reshaped into image dimensions)
        # window: descpeckling filter window (tuple)
        # default noise variance = 0.25
        # assumes noise mean = 0

        mean_window = uniform_filter(band, window)
        mean_sqr_window = uniform_filter(band**2, window)
        var_window = mean_sqr_window - mean_window**2
        SD_window = np.sqrt(var_window);


        C_U = 1/(np.sqrt(numL)*var_noise)
        C_max = np.sqrt(1+2/numL)
        C_L = SD_window/mean_window
        K = np.exp(-damp*(C_L - C_U)/(C_L - C_max))

        if (C_L <= C_U):
            band_filtered = mean_window
        elif(C_L > C_U and C_L < C_max):
            band_filtered = mean_window*K + (1-K)*band
        else:
            band_filtered = band;

        return band_filtered



img[:,:,0] = decibel_to_linear(img[:,:,0]);
img[:,:,1] = decibel_to_linear(img[:,:,1]);
noise_var_1 = np.round(np.var(img[:,:,0])*var,10)
noise_var_2 = np.round(np.var(img[:,:,1])*var,10)




img[:,:,0] = lee_filter(img[:,:,0], window, noise_var_1)
img[:,:,1] = lee_filter(img[:,:,1], window, noise_var_2)
