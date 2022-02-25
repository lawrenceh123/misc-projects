import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
import copy
from scipy import interpolate
import scipy.stats as stats

def get_subpixel(signal):
    oldx = np.arange(len(signal))
    oldy = signal
    f = interpolate.interp1d(oldx, oldy)
    newx = np.arange(0, len(signal)-0.9, 0.1)
    newy = f(newx)
    return newx, newy

def get_interp_crossings(signal, crossings, local_diff):
    newx, newy = get_subpixel(signal)
    
    result = ([aa for aa in newx[np.flatnonzero(np.diff(np.sign(newy)))]
               if aa >= crossings[np.argmax(local_diff)] and aa <= crossings[np.argmax(local_diff)]+1][0],
              [aa for aa in newx[np.flatnonzero(np.diff(np.sign(newy)))]
               if aa >= crossings[np.argmin(local_diff)] and aa <= crossings[np.argmin(local_diff)]+1][0])
    
    return np.round(result, 1)

def laplace_edge_detector(signal, scale_signal, filt_type, filt_param, 
                          do_plot=True, thr1val=50, thr2val=50, verbose=True):
    
    if scale_signal:
        try:
            signal = MinMaxScaler().fit_transform(signal.to_numpy().reshape(-1, 1)).flatten()
        except:
            signal = MinMaxScaler().fit_transform(signal.reshape(-1, 1)).flatten()
    
    if filt_type == 'gaussian':
        signal_filt = gaussian_filter1d(signal, sigma=filt_param)
    elif filt_type == 'savgol':
        signal_filt = savgol_filter(signal, window_length=filt_param, polyorder=filt_param-2)
        # signal_filt = savgol_filter(signal, window_length=filt_param, polyorder=filt_param-4)
    else:
        signal_filt = signal
        if verbose: print('no filtering')
    
    signal_laplace = convolve(signal_filt, [1, -2, 1], mode='same', method='auto')
    crossings = np.flatnonzero(np.diff(np.sign(signal_laplace)))
    if verbose: print(crossings)
    # exclude edge effects
    # crossings = [xx for xx in crossings if xx!= 0 and xx != len(signal)-2] 
    crossings = [xx for xx in crossings if (xx > 0) & (xx < len(signal)-2)]
    # -1 for 0 index, -1 for diff
    
    if verbose: print(crossings)
    # filter out median 50% x values, keeping first and last elements
    # thr1val, thr2val = 50, 50
    thr1 = np.percentile(np.arange(len(signal)), thr1val).round()
    thr2 = np.percentile(np.arange(len(signal)), thr2val).round()
    middle_elements_filt = [xx for xx in crossings[1:-1] if (xx <= thr1) or (xx >= thr2)]
    crossings2 = []
    crossings2.append(crossings[0])
    crossings2.extend(middle_elements_filt)
    crossings2.append(crossings[-1])

    local_diff = [signal_filt[crossing+1]-signal_filt[crossing-1] for crossing in crossings]
    local_diff2 = [signal_filt[crossing+1]-signal_filt[crossing-1] for crossing in crossings2]
       
    # crossings_filt = crossings[np.argmax(local_diff)], crossings[np.argmin(local_diff)]
    crossings_filt2 = crossings2[np.argmax(local_diff2)], crossings2[np.argmin(local_diff2)]
    
    # check for negative
    if np.argmax(local_diff2) >= np.argmin(local_diff2):
        # raise ValueError('Zero or negative diameter??')
        print('line profile may be irregular...skipping...')
        return {'raw':np.nan, 'interp':np.nan}
    
    crossings_filt2_interp = get_interp_crossings(signal=signal_laplace, 
                                                  crossings=crossings2, local_diff=local_diff2)
    distance = np.diff(crossings_filt2)[0]
    distance_interp = np.round(np.diff(crossings_filt2_interp)[0], 1)

    if do_plot:
        fig, (ax, ax2) = plt.subplots(2,1, figsize=(8,6), gridspec_kw={'height_ratios': [1, 1]}, sharex=True)
        ax.plot(signal, 'k', lw=2, alpha=1, label='Signal')
        #ax.plot(signal_filt, lw=2, alpha=0.8, label='filtered')
        ax1 = ax.twinx()
        signal_laplace_plot = copy.deepcopy(signal_laplace)
        signal_laplace_plot[0] = np.nan
        signal_laplace_plot[-1] = np.nan
        ax1.plot(signal_laplace_plot, c='darkgray', lw=2, alpha=1, label='Laplacian')
        ax1.axhline(0, c='k', ls=':')
        ax1.plot(crossings, [0]*len(crossings), 'o', c='darkgray')

        ax1.axvline(crossings_filt2_interp[0], c='k', ls=':')
        ax1.axvline(crossings_filt2_interp[-1], c='k', ls=':')
        ax1.set_ylabel('Laplacian', color='darkgray')
        ax1.tick_params(axis='y', labelcolor='darkgray')

        ax2.plot(crossings2, local_diff2, 'o', c='darkgray', label='Zero-crossings')
        ax2.plot(crossings, local_diff, 'o', c='darkgray', alpha=0.25)
        
        ax1.plot(crossings_filt2[0], 0, 'o', mec='b', mew=2, mfc='none')
        ax1.plot(crossings_filt2[1], 0, 'o', mec='b', mew=2, mfc='none')
        ax2.plot(crossings_filt2[0], max(local_diff2), 'o', mec='b', mew=2, mfc='none', label='Detected edges')
        ax2.plot(crossings_filt2[1], min(local_diff2), 'o', mec='b', mew=2, mfc='none')
        
        ax2.grid()

        ax.set(ylabel='Luminance (AU)', xlabel='')
        ax2.set(xlabel='Position along vessel (pixel)', ylabel='Change in luminance\nat zero-crossings')
        ax2.legend(fontsize=12)
        ax.text(10, 0, f'Interpolated distance =\n{distance_interp} pixels', ha='center', fontsize=12)

        fig.tight_layout()
        plt.show()
    
    return {'raw':distance, 'interp':distance_interp}