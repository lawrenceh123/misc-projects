from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy
from scipy.io import loadmat
import pickle
import random

def plot_mean_var_time(photon_flux_txy, perc_min_flux, perc_max_flux):
    # mean and var through time
    arr_m = np.nanmean(photon_flux_txy, axis=0).ravel()
    arr_v = np.nanvar(photon_flux_txy, axis=0).ravel()
    
    arr_m = arr_m[~np.isnan(arr_m)]
    arr_v = arr_v[~np.isnan(arr_v)]
    
    # plot results
    m_scale = np.percentile(arr_m, [perc_min_flux, perc_max_flux])
    v_scale = np.percentile(arr_v, [perc_min_flux, perc_max_flux])
    keep = (arr_m>=m_scale[0]) & (arr_m<=m_scale[1]) & (arr_v>=v_scale[0]) & (arr_v<=v_scale[1])
    
    arr_m = arr_m[keep]
    arr_v = arr_v[keep]
        
    fig, ax = plt.subplots(figsize=(3.5,3.5))
    ax.scatter(arr_m, arr_v, s=5, c='k', alpha=0.5)
    ll = np.linspace(min(np.min(arr_v), np.min(arr_m)), max(np.max(arr_v), np.max(arr_m)))
    ax.plot(ll,ll,'darkgray', linestyle=':', label='mean=var') 
    
    # fit
    p = np.polyfit(arr_m, arr_v, deg=1)
    ax.plot(arr_m, p[0]*arr_m+p[1], 'r', label='fit')
    
    ax.set_yticks(ax.get_xticks())
    ax.set_xticks(ax.get_yticks())
    ax.set_xlim(min(np.min(arr_v), np.min(arr_m)), max(np.max(arr_v), np.max(arr_m)))
    ax.set_ylim(min(np.min(arr_v), np.min(arr_m)), max(np.max(arr_v), np.max(arr_m)))
    ax.set_xlabel('Mean\n# photons over time')
    ax.set_ylabel('Variance\n# photons over time')
    ax.set_aspect('equal')
    ax.legend(fontsize=10)
#    ax.set_title('tid: {}\nperc_min:{}, perc_max:{}\nmean of mean: {:.1f}'.format(tid, perc_min_flux, perc_max_flux, arr_m.mean()))
#    ax.set_title('Photon flux per dwell time')
    plt.grid()
    plt.tight_layout()
    plt.show()
    
    return {'fig':fig, 'fit':p}

def get_photon_gain_parameters2(vv, crop=(0,0), perc_min=3, perc_max=90):
    # modified from Jerome to use with variable gain across x-axis
    # Remove saturated pixels
    idxs_not_saturated = np.where(vv.max(axis=0).flatten() < 8191) #2**13
    
    _var = vv.var(axis=0).flatten()[idxs_not_saturated]
    _mean = vv.mean(axis=0).flatten()[idxs_not_saturated]
    
    # Remove pixels that deviate from Poisson stats
#    perc_min, perc_max = 3, 90
    _var_scale = np.percentile(_var, [perc_min, perc_max])
    _mean_scale = np.percentile(_mean, [perc_min, perc_max])
    
    # Remove outliers
    _var_bool = np.logical_and(_var > _var_scale[0], _var < _var_scale[1])
    _mean_bool = np.logical_and(_mean > _mean_scale[0], _mean < _mean_scale[1])
    _no_outliers = np.logical_and(_var_bool, _mean_bool)
    
    _var_filt = _var[_no_outliers]
    _mean_filt = _mean[_no_outliers]
    _mat = np.vstack([_mean_filt, np.ones(len(_mean_filt))]).T
    try:
        slope, offset = np.linalg.lstsq(_mat, _var_filt, rcond=None)[0]
    except np.linalg.LinAlgError:
        raise DataCorruptionError('Unable to get photon metrics - check video for anomalies.')
    
    return {'var': _var_filt,
            'mean': _mean_filt,
            'slope': slope,
            'offset': offset}

# color legend for other use
def oephys_legend():
    fig, ax = plt.subplots(figsize=(4,4))
    ax.plot([1,2],[1,2],c=(0,0,0.85), label='Emx1-s',lw=3)
    ax.plot([1,2],[1,2],c=(0.2,0.2,0.2), label='tetO-s',lw=3)
    ax.plot([1,2],[1,2],c=(0.85,0,0), label='Emx1-f',lw=3)
    ax.plot([1,2],[1,2],c=(0,0.5,0), label='Cux2-f',lw=3)
    plt.show()
    
    fig2, ax2 = plt.subplots(figsize=(1,1))
    ax2.legend(*ax.get_legend_handles_labels(), loc='center', fontsize=14)
    ax2.axis('off')
    fig2.tight_layout()
    plt.show()
    #fig2.savefig('plots/mylegend.pdf',format='pdf', dpi=300, bbox_inches='tight')    

def dbl_exp(x, tauR, tauD, amp):
    '''
    Jack's function for fitting double exponential
    '''
    return amp * (1 - np.exp(-x/tauR)) * np.exp(-x/tauD)

def fit_DFF(tt, mean_DFF, gtype):
    '''
    Fitting mean dF/F from Jack
    Fit between t=0 and t= 1 s.
    Check resulting tauR and tau D.
    If tauR or tauD is near maximum value, fit between t=0 and t=05.
    If tauR or tauD is near maximum value, return an error warning.    
    '''

    tauR_max = 1 # max limit in seconds.
    tauD_max = 3 # max limit in seconds.
    
    x_start = 0
    
    if gtype=='Emx1-s' or gtype=='tetO-s':
        x_end = 1
    else:
        x_end = 0.5
    
    fit_data = mean_DFF[(tt>=x_start) & (tt<=x_end)]
    fit_time = tt[(tt>=x_start) & (tt<=x_end)]
    
    popt, pcov = scipy.optimize.curve_fit(dbl_exp, fit_time, fit_data, bounds=([0, 0, 0], 
                                                                               [tauR_max, tauD_max, np.inf]))
    tauR = popt[0]
    tauD = popt[1]
    amp = popt[2]
    fit = dbl_exp(fit_time, tauR, tauD, amp)
    
    if (tauR > 0.9*tauR_max) or (tauD > 0.9*tauD_max):
        if gtype=='Emx1-s' or gtype=='tetO-s':
            x_end = 0.5
        else:
            x_end = 0.2
        
        fit_data = mean_DFF[(tt>=x_start) & (tt<=x_end)]
        fit_time = tt[(tt>=x_start) & (tt<=x_end)]
        popt, pcov = scipy.optimize.curve_fit(dbl_exp, fit_time, fit_data, bounds=([0, 0, 0], 
                                                                               [tauR_max, tauD_max, np.inf]))
        tauR = popt[0]
        tauD = popt[1]
        amp = popt[2]
        fit = dbl_exp(fit_time, tauR, tauD, amp)
    
    if (tauR > 0.9*tauR_max) or (tauD > 0.9*tauD_max):
        error = True
    else:
        error = False
    
    return tauR, tauD, amp, fit_time, fit, error

def get_nspike_events(ispk, iframes, wspk_pre, wspk, wspk_post, dte, verbose, do_plots, vmfd):
    '''
    Parameters:
    ispk: spike idx 
    iframes: correspondence between ophys & ephys
    wspk_pre: spike window, pre
    wspk: spike window, summation
    wspk_post: spike window, post
    dte: sampling rate for ephys
    vmfd: Vm trace

    Returns:
    Time of first spike and number of spike in isolated spiking events 
    '''

    # discard spikes that occurred in the first 10 ophys frames
    tidx = ispk[ispk>iframes[9]]
    temp = np.zeros(iframes[-1])
    temp[ispk] = 1


    # count=0
    espk, tspk = [], []
    # while count<11:
    while len(tidx):
        if verbose:
            print(f'checking tidx {tidx[0]}')
        # count+=1
        if np.sum(temp[int(tidx[0]-np.round(wspk_pre/dte)):tidx[0]])==0 and tidx[0]+np.round(wspk_post/dte)<=len(temp):
            # no spikes in pre-spike window and tidx+post-spike window does not exceed length of ophys
            if np.sum(temp[tidx[0]:int(tidx[0]+np.round(wspk/dte))])>1: # spiking event has more than 1 spike
                last_spike = tidx[int(np.sum(temp[tidx[0]:int(tidx[0]+np.round(wspk/dte))])-1)] # find last spike of spiking event
            else:
                last_spike = tidx[0]
            
            if last_spike+np.round(wspk_post/dte)<=len(temp) and np.sum(temp[last_spike:int(last_spike+np.round(wspk_post/dte))])==1:
                # last spike+post-spike window does not exceed length of ophys,and no spikes in the post-spike window (the only spike is last_spike)'
                espk.append(np.sum(temp[tidx[0]:int(tidx[0]+np.round(wspk/dte))]))
                tspk.append(tidx[0])
                if verbose:
                    print('analyze this event')
                if do_plots:
                    fig, ax = plt.subplots(figsize=(15,2))
                    tt = np.arange(-np.round(wspk_pre/dte), last_spike-tidx[0]+np.round(wspk_post/dte))
                    ax.plot(tt*dte, vmfd[int(tidx[0]-np.round(wspk_pre/dte)):int(last_spike+np.round(wspk_post/dte))])
                    yl = ax.get_ylim()                    
                    ax.plot([-wspk_pre, -wspk_pre], [yl[0], yl[1]], '--k', alpha=0.9)
                    ax.plot([0, 0], [yl[0], yl[1]], ':k', alpha=0.9)
                    ax.plot([wspk, wspk], [yl[0], yl[1]], ':k', alpha=0.9)
                    ax.plot([(last_spike-tidx[0])*dte+wspk_post, (last_spike-tidx[0])*dte+wspk_post], [yl[0], yl[1]], '--k', alpha=0.9)
                    ax.set_title(f'tidx: {tidx[0]}, spikes: {np.sum(temp[tidx[0]:int(tidx[0]+np.round(wspk/dte))])}')
                    plt.show()
            else:
                if verbose:
                    print('skip this event (post)')
                if do_plots:
                    fig, ax = plt.subplots(figsize=(15,2))
                    tt = np.arange(-np.round(wspk_pre/dte), last_spike-tidx[0]+np.round(wspk_post/dte))
                    ax.plot(tt*dte, vmfd[int(tidx[0]-np.round(wspk_pre/dte)):int(last_spike+np.round(wspk_post/dte))], 'r')
                    yl = ax.get_ylim()
                    ax.plot([-wspk_pre, -wspk_pre], [yl[0], yl[1]], '--k', alpha=0.9)
                    ax.plot([0, 0], [yl[0], yl[1]], ':k', alpha=0.9)
                    ax.plot([wspk, wspk], [yl[0], yl[1]], ':k', alpha=0.9)
                    ax.plot([(last_spike-tidx[0])*dte+wspk_post, (last_spike-tidx[0])*dte+wspk_post], [yl[0], yl[1]], '--k', alpha=0.9)
                    ax.set_title(f'tidx: {tidx[0]}, post')
                    plt.show()
                
            tidx = tidx[tidx>last_spike] # no need to plus post-spike window, because will check for pre-spike window during next event
        else:
            if verbose:
                print('skip this event (pre)')
            if do_plots:
                fig, ax = plt.subplots(figsize=(15,2))
                tt = np.arange(-np.round(wspk_pre/dte), np.round(wspk_post/dte))
                ax.plot(tt*dte, vmfd[int(tidx[0]-np.round(wspk_pre/dte)):int(tidx[0]+np.round(wspk_post/dte))], 'orange')
                yl = ax.get_ylim()
                ax.plot([-wspk_pre, -wspk_pre], [yl[0], yl[1]], '--k', alpha=0.9)
                ax.plot([0, 0], [yl[0], yl[1]], ':k', alpha=0.9)
                ax.plot([wspk, wspk], [yl[0], yl[1]], ':k', alpha=0.9)
                ax.set_title(f'tidx: {tidx[0]}, pre')
                plt.show()
            tidx = tidx[1:]

    tspk = np.array(tspk)
    espk = np.array(espk)
    return (tspk, espk.astype(int))

def get_zerospike_idx(tid, ispk, iframes, dto, dte, fmean, wlen=1, wpre=4, wpost=1):
    # find no spike window(s) of >= wlen s, separated by >= [wpre, wpost] s
    # wlen = 1
    # wpre = 4
    # wpost = 1
    wtot = wlen+wpre+wpost
    
    # spikes that need to be considered
    tspk0 = ispk[ispk>iframes[0]]
    tspk0 = tspk0[tspk0<iframes[-1]]  # added for downsampled dff

    assert len(tspk0)==len(np.unique(tspk0)) 
    isi = np.diff(tspk0)*dte
    ee1 = tspk0[np.where(isi>=wtot)[0]]
    ee2 = tspk0[np.where(isi>=wtot)[0]+1]
    
    # edge cases
    # 1st spike
    if np.where(iframes>=tspk0[0])[0][0]*dto>=wtot:
        ee1 = np.insert(arr=ee1, obj=0, values=iframes[0])
        ee2 = np.insert(arr=ee2, obj=0, values=tspk0[0])
    # last spike
    if (len(iframes)-np.where(iframes>=tspk0[-1])[0][0])*dto>=wtot:
        ee1 = np.insert(arr=ee1, obj=len(ee1), values=tspk0[-1])
        ee2 = np.insert(arr=ee2, obj=len(ee2), values=iframes[-1])
    
    # special cases
    # if tid==103721:
    #     ee2 = ee2[ee1<=np.round(160/dte)]
    #     ee1 = ee1[ee1<=np.round(160/dte)]
    # if tid==103727:
    #     ee2 = ee2[ee1<=np.round(120/dte)]
    #     ee1 = ee1[ee1<=np.round(120/dte)]
    # if tid==102865:
    #     ee2 = ee2[ee1<=np.round(40/dte)]
    #     ee1 = ee1[ee1<=np.round(40/dte)]
    # if tid==103406:
    #     ee2 = ee2[ee1<=np.round(222/dte)]
    #     ee1 = ee1[ee1<=np.round(222/dte)]
    # if tid==103467:
    #     ee2 = ee2[ee1<=np.round(215/dte)]
    #     ee1 = ee1[ee1<=np.round(215/dte)]
    
    assert len(ee1)==len(ee2) 
    
    temp = np.zeros(iframes[-1]-iframes[0])
    temp[tspk0-iframes[0]] = 1
    fig, ax = plt.subplots(figsize=(12,3))
    ax1 = ax.twinx()
    ax1.plot(np.arange(len(temp))*dte, temp, 'r', lw=0.5)
    ax.plot(np.arange(len(fmean))*dto, fmean, 'k', lw=1)
    ax1.set_ylim(top=5)
    
    for ii in range(len(ee1)):
        rect_w = (np.where(iframes>=ee2[ii]-wpost/dte)[0][0]-np.where(iframes>=ee1[ii]+wpre/dte)[0][0])*dto
        rect_h = ax1.get_ylim()[1]-ax1.get_ylim()[0]
        rect_x = (np.where(iframes>=ee1[ii]+wpre/dte)[0][0])*dto
        rect_y = ax1.get_ylim()[0]
        # rect = patches.Rectangle( ((ee1[ii]-iframes[0])*dte, 0), (ee2[ii]-ee1[ii])*dte, 5, linestyle='--', linewidth=1, edgecolor='darkgrey', facecolor='none')
        rect = patches.Rectangle( (rect_x, rect_y), rect_w, rect_h, linestyle='--', linewidth=1, edgecolor='darkgrey', facecolor='none')
        ax1.add_patch(rect)
    
    for aa in (ax, ax1):
        aa.spines['top'].set_visible(False)
        aa.spines['right'].set_visible(False)
    ax.set_ylabel('Fluorescence (A.U.)')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'{tid}')
    ax1.set_yticks([])
    fig.tight_layout()
    plt.show()
    
    # start and stop of no-spike windows
    nidx_start, nidx_stop = [], []
    for ii in range(len(ee1)):
        nidx_start.append(np.where(iframes>=ee1[ii]+wpre/dte)[0][0])
        nidx_stop.append(np.where(iframes>=ee2[ii]-wpost/dte)[0][0])
    
    if not len(nidx_start):
        print('no zero-spike windows found')
    
    return {'nidx_start':nidx_start, 'nidx_stop':nidx_stop}


def get_preprocessed(tid, dn):
    # load oephys data
    matfile = loadmat(dn+str(tid)+'_oe.mat', variable_names=['iFrames', 'dto', 'dte', 'ROI_list', 'e'])
    # oephys sync
    iframes = matfile['iFrames'].ravel()
    iframes-=1 # change to 0 based
    # ophys
    dto = matfile['dto'].ravel()[0]
    roi_list = matfile['ROI_list']
    roi_list_index = list(roi_list.dtype.names)
    pixel_list = roi_list[0][0][roi_list_index.index('pixel_list')].ravel()
    fmean = roi_list[0][0][roi_list_index.index('fmean')].ravel() # raw mean cell trace
    fnp = roi_list[0][0][roi_list_index.index('fmean_np')].ravel()
    # ephys
    dte = matfile['dte'].ravel()[0]
    ee = matfile['e']
    ee_index = list(ee.dtype.names)
    ispk = ee[0][0][ee_index.index('iSpk')].ravel()
    ispk-=1 # change to 0 based
    
    # special cases
    # if tid==102865:
    #     ispk = ispk[:36]
    #     print('special case, some ispk not analyzed')
    if tid==103406:
        ispk = ispk[ispk<=222/dte]
        print(f'special case {tid}, some ispk not analyzed')
    if tid==103467:
        ispk = ispk[ispk<=215/dte]
        print(f'special case {tid}, some ispk not analyzed')

    spk = ee[0][0][ee_index.index('spk')].ravel()
    vmfd = ee[0][0][ee_index.index('Vmfd')].ravel()
    vm = ee[0][0][ee_index.index('Vm')].ravel()
    
    return {'iframes':iframes, 'dto':dto, 'roi_list':roi_list, 'pixel_list':pixel_list, 'fmean':fmean, 'fnp':fnp,
            'dte':dte, 'ee':ee, 'ispk':ispk, 'spk':spk, 'vmfd':vmfd, 'vm':vm}


def prepare_spike_detection(tid, num_spike, rval, preproc_dn, caseg_fn, nidx_fn, do_plots, min_nsegs=25, seed=0):
    random.seed(seed)
    with open(caseg_fn, 'rb') as f:
        data = pickle.load(f)[tid, rval]
    try:
        casegs = data['caseg_dict_wca'][num_spike]
    except KeyError as ke:
        print(f'{type(ke)}: no {ke}-spike events')
        return (np.empty((0,0)), np.empty((0,0)))
    dff_pre = data['window_dict']['spk_pre']
    dff_post = data['window_dict']['spk_post']
    
    preproc = get_preprocessed(tid, preproc_dn)
    dto = preproc['dto']
    fmean = preproc['fmean']
    fnp = preproc['fnp']
    fmean = fmean-rval*fnp
    
    with open(nidx_fn, 'rb') as f:
        nidx = pickle.load(f)[tid]
    nidx_start = nidx['nidx_start']
    nidx_stop = nidx['nidx_stop']
    
    # match # noise with # spiking events, min N events
    nidx_list = []
    for ii in range(len(nidx_start)):
        for jj in range(int(np.ceil(max(casegs.shape[0], min_nsegs)/len(nidx_start)))):        
            nidx_list.append(random.randint(nidx_start[ii]+np.round(dff_pre/dto), nidx_stop[ii]-np.round(dff_post/dto)))
    
    tt = np.arange(-np.round(dff_pre/dto), np.round(dff_post/dto)+1)*dto
    nsegs = np.empty((0, len(tt))) 
    for xx in nidx_list:
        nseg = fmean[int(xx-np.round(dff_pre/dto)):int(xx+np.round(dff_post/dto))+1]
        nsegs = np.vstack([nsegs, nseg])
    
    if nsegs.size:
        if do_plots:
            fig, (ax, ax1) = plt.subplots(1,2, sharex=True, sharey=False, figsize=(8,4))
            ax.plot(tt, nsegs.T, 'r', alpha=0.25)
            ax.plot(tt, casegs.T, 'b', alpha=0.25)
            ax1.plot(tt, nsegs.mean(axis=0), 'r', alpha=1, lw=3, label='0-spike')
            ax1.plot(tt, casegs.mean(axis=0), 'b', alpha=1, lw=3, label=f'{num_spike}-spike')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('F (AU)')
            ax1.set_xlabel('Time (s)')
            ax.set_title(f'tid:{tid}, rval:{rval}')
            ax1.legend()
            fig.tight_layout
            plt.show()
        else:
            print(f'tid: {tid}, rval: {rval}, seed: {seed}, preparing traces')
            # pass
    else:
        print(f'{tid}: no zero-spike windows found')
    
    return (casegs, nsegs)

def get_tpr(casegs, nsegs, do_plots):
    template = np.mean(casegs, axis=0)    
    unit = (template-np.mean(template))/np.linalg.norm(template-np.mean(template))
    
    # if do_plots:
    #     fig, ax = plt.subplots()
    #     ax.plot(unit,'k')
    #     ax.set_title('unit')
    #     ax.set_ylabel('F')
    #     ax.set_xlabel('sample #')
    #     fig.tight_layout()
    #     plt.show()
    
    si, ni = [],[] # signal and noise
    for xx in range(casegs.shape[0]):
        si.append(np.dot(casegs[xx, :], unit)/np.linalg.norm(unit))
    for xx in range(nsegs.shape[0]):
        ni.append(np.dot(nsegs[xx, :], unit)/np.linalg.norm(unit))
    
    tprs = np.array([])
    tps = np.array([]) 
    for vv in range(101):    
        thr = np.percentile(ni, vv)   
        tpr = np.sum(si > thr)/len(si)
        tprs = np.append(tprs, tpr)
        tps = np.append(tps, np.sum(si > thr))
    tprs = np.append(tprs, 0) # connect to origin for plotting
    
    # fpr with extra 0 for plotting
    fprs = 1-np.arange(0,1.01,0.01)
    fprs = np.append(fprs, 0)    

    if do_plots:
        bins = np.linspace(min(min(ni), min(si)), max(max(ni), max(si)), 20)
        fig, (ax, ax1) = plt.subplots(1,2, figsize=(8,4))
        ax.hist(si,bins=bins,color='b', alpha=0.5,label='signal')
        ax.hist(ni,bins=bins,color='r', alpha=0.5, label='noise')
        ax.set_ylabel('count')
        ax.set_xlabel('scalar projection of signal and noise onto unit')
        ax.set_title('scalar projection of signal and noise onto unit\n dot(signal, unit)/norm(unit)\ndot(noise, unit)/norm(unit)')
        ax.legend()
        # fig.tight_layout()
        # plt.show()
                
        # fig, ax = plt.subplots(figsize=(4,4))
        ax1.plot(fprs, tprs.T, 'k', alpha=0.5)
        ax1.set_ylim(-0.05,1.05)
        ax1.set_xlim(-0.05,1.05)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_ylabel('True positive rate')
        ax1.set_xlabel('False positive rate')
        labels = np.round(np.arange(0,1.2,0.2),1)
        ax1.set_xticks(labels) # plt.xticks(labels, labels)
        ax1.set_yticks(labels) # plt.yticks(labels, labels)
        ax1.set_aspect('equal')
        fig.tight_layout()
        plt.show()
    else:
        print('calculating tpr')
    
    return (tprs, fprs, tps, len(si))

def get_acid(print_n=False):
    df = pd.read_excel('new_tids.xlsx', header=None)
    df.dropna(axis=0, how='all', inplace=True)
    df.columns = ['gtype', 'aid', 'cid', 'tid']
    
    df.fillna(method='ffill', inplace=True)
    df[['aid', 'cid', 'tid']] = df[['aid', 'cid', 'tid']].astype(int)
    df['acid'] = df[['aid','cid']].apply(tuple, axis=1)
    
    if print_n:
        print('final high-zoom dataset: {} recordings from {} cells from {} mice'.format(
            df['tid'].nunique(), df['acid'].nunique(), df['aid'].nunique()))
    return df

    # df.to_pickle('new_tids_acid.pickle')

def check_spikes_dFF_traces(tid, fn='new_tids_all_casegs_new_optimal_rval.pickle'):
    '''
    check spikes and dF/F
    need vmfdsegs, casegs, dffsegs
    ''' 
    with open(fn, 'rb') as f:
        data = pickle.load(f)

    try:
        rval = [k[1] for k in data if k[0]==tid][0]
        
        vmfdseg_dict = data[tid, rval]['vmfdseg_dict']
        caseg_dict = data[tid, rval]['caseg_dict']
        dffseg_dict = data[tid, rval]['dffseg_dict']
        tspk_dict = data[tid, rval]['tspk_dict']
        
        wspk_pre = data[tid, rval]['window_dict']['wspk_pre']
        wspk_post = data[tid, rval]['window_dict']['wspk_post']
        dff_pre = data[tid, rval]['window_dict']['dff_pre']
        dff_post = data[tid, rval]['window_dict']['dff_post']

    
        max_pre = max(wspk_pre, dff_pre)
        max_post = max((wspk_post, dff_post))
    
        preproc = get_preprocessed(tid, 'oephys2/')
        dte = preproc['dte']
        dto = preproc['dto']
    
        tt_caseg = np.arange(-np.round(max_pre/dto), np.round(max_post/dto)+1)*dto
        tt_caseg_e = np.arange(-np.round(max_pre/dte), np.round(max_post/dte)+1)*dte
        
        mycmap = dict(zip(np.arange(1,6), plt.rcParams['axes.prop_cycle'].by_key()['color'][:5]))
        plt.rcParams.update({'font.size': 10})   
        for kk in dffseg_dict:
            for ll in range(vmfdseg_dict[kk].shape[0]):
                fig, (ax, ax1) = plt.subplots(2,1, figsize=(16,6), sharex=True)
                ax.plot(tt_caseg_e, vmfdseg_dict[kk][ll, :], color=mycmap[kk])
                ax1.plot(tt_caseg, dffseg_dict[kk][ll, :], color=mycmap[kk])
                ax2 = ax1.twinx()
                ax2.plot(tt_caseg, caseg_dict[kk][ll, :], color=mycmap[kk])
                
                ax.set_xlim(-dff_pre, dff_post)
                ax1.set_xlim(-dff_pre, dff_post)
                ax.set_title(f'tidx: {tspk_dict[kk][ll]}, #spikes: {kk}')
                ax1.set_title(f'rval: {rval}')
                ax1.set_ylabel('dF/F')
                ax1.set_xlabel('Time after spike (s)')
                ax2.set_ylabel('F')
                plt.show()
    except IndexError as ie:
        print(ie, '; tid not found.')

def vstack_dicts(source_dict_list):
    ''' stack np arrays (dffsegs) in list of dictionaries '''

    # source = [{1:np.array([[1,2,3], [4,5,6]]),2:np.array([[1,2,3], [1,2,3]])}, 
    #           {1:np.array([[7,8,9], [7,8,9]]),2:np.array([[11,12,13], [11,12,13]])}]
    result = {}
    all_keys = set().union(*(d.keys() for d in source_dict_list)) # find unique keys in list of dictionaries
    for key in all_keys:
        result[key] = np.vstack([d.get(key) for d in source_dict_list if d.get(key) is not None])
    return result

def prepare_spike_detection_ds(tid, num_spike, rval, preproc_dn, caseg_fn, nidx_fn, do_plots, min_nsegs=25, seed=0):
    random.seed(seed)
    with open(caseg_fn, 'rb') as f:
        data = pickle.load(f)[tid, rval]
    try:
        casegs = data['caseg_dict_wca'][num_spike]
    except KeyError as ke:
        print(f'{type(ke)}: no {ke}-spike events')
        return (np.empty((0,0)), np.empty((0,0)))
    dff_pre = data['window_dict']['spk_pre']
    dff_post = data['window_dict']['spk_post']
    
    data = [xx for xx in preproc_dn if xx[0].endswith(str(tid)+'.h5')][0]        
    fmean = data[2].flatten()
    dto = 1/30
    
    with open(nidx_fn, 'rb') as f:
        nidx = pickle.load(f)[tid]
    nidx_start = nidx['nidx_start']
    nidx_stop = nidx['nidx_stop']
    
    # match # noise with # spiking events, min N events
    nidx_list = []
    for ii in range(len(nidx_start)):
        for jj in range(int(np.ceil(max(casegs.shape[0], min_nsegs)/len(nidx_start)))):        
            nidx_list.append(random.randint(nidx_start[ii]+np.round(dff_pre/dto), nidx_stop[ii]-np.round(dff_post/dto)))
    
    tt = np.arange(-np.round(dff_pre/dto), np.round(dff_post/dto)+1)*dto
    nsegs = np.empty((0, len(tt))) 
    for xx in nidx_list:
        nseg = fmean[int(xx-np.round(dff_pre/dto)):int(xx+np.round(dff_post/dto))+1]
        nsegs = np.vstack([nsegs, nseg])
    
    if nsegs.size:
        if do_plots:
            fig, (ax, ax1) = plt.subplots(1,2, sharex=True, sharey=False, figsize=(8,4))
            ax.plot(tt, nsegs.T, 'r', alpha=0.25)
            ax.plot(tt, casegs.T, 'b', alpha=0.25)
            ax1.plot(tt, nsegs.mean(axis=0), 'r', alpha=1, lw=3, label='0-spike')
            ax1.plot(tt, casegs.mean(axis=0), 'b', alpha=1, lw=3, label=f'{num_spike}-spike')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('F (AU)')
            ax1.set_xlabel('Time (s)')
            ax.set_title(f'tid:{tid}, rval:{rval}')
            ax1.legend()
            fig.tight_layout
            plt.show()
        else:
            print(f'tid: {tid}, rval: {rval}, seed: {seed}, preparing traces')
            # pass
    else:
        print(f'{tid}: no zero-spike windows found')
    
    return (casegs, nsegs)

