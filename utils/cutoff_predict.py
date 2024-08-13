import numpy as np
from scipy.stats import kurtosis, skew
import xgboost as xgb
import os
import sys

def feature_extraction(nexp, nsub, nfoci):
    nsub_total = np.sum(nsub)
    nsub_mean = np.mean(nsub)
    nsub_median = np.median(nsub)
    nsub_std = np.std(nsub)
    nsub_max = np.max(nsub)
    if nsub_max > 300:
        print('Dataset features parameters that would lead to Out-Of-Distribution prediction: Accuracy can\'t be guaranteed. Please disable cutoff prediction!')
        sys.exit()
    nsub_min = np.min(nsub)
    nsub_skew = skew(nsub)
    nsub_kurtosis = kurtosis(nsub)
    
    nfoci_total = np.sum(nfoci)
    nfoci_mean = np.mean(nfoci)
    nfoci_median = np.median(nfoci)
    nfoci_std = np.std(nfoci)
    nfoci_max = np.max(nfoci)
    if nfoci_max > 150:
        print('Dataset features parameters that would lead to Out-Of-Distribution cutoff prediction prediction: Accuracy can\'t be guaranteed. Please disable cutoff prediction!')
        sys.exit()
    nfoci_min = np.min(nfoci)
    nfoci_skew = skew(nfoci)
    nfoci_kurtosis = kurtosis(nfoci)
    
    ratio_mean = np.mean(nfoci / nsub)
    ratio_std = np.std(nfoci / nsub)
    ratio_max = np.max(nfoci / nsub)
    ratio_min = np.min(nfoci / nsub)
    
    nstudies_foci_ratio = nfoci_total / nexp
    
    hi_foci = 0
    mi_foci = 0
    li_foci = 0
    vi_foci = 0
    
    for i in range(nexp):
        if nsub[i] > 20:
            hi_foci += nfoci[i]
        if (nsub[i] < 20) and (nsub[i] > 15):
            mi_foci += nfoci[i]
        if (nsub[i] < 15) and (nsub[i] > 10):
            li_foci += nfoci[i]
        if nsub[i] < 10:
            vi_foci += nfoci[i]
    
    
    x = np.c_[nexp,
              nsub_total, nsub_mean, nsub_median, nsub_std, nsub_max, nsub_min, nsub_skew, nsub_kurtosis,
              nfoci_total, nfoci_mean, nfoci_median, nfoci_std, nfoci_max, nfoci_min, nfoci_skew, nfoci_kurtosis,
              ratio_mean, ratio_std, ratio_max, ratio_min, nstudies_foci_ratio,
              hi_foci, mi_foci, li_foci, vi_foci]
    
    return x

def predict_cutoff(exp_df):
    path = os.path.abspath(__file__)
    xgb_vfwe = xgb.XGBRegressor()
    xgb_vfwe.load_model(f'{path[:-18]}/ml_models/vFWE_model.txt')

    xgb_cfwe = xgb.XGBRegressor()
    xgb_cfwe.load_model(f'{path[:-18]}/ml_models/cFWE_model.txt')

    xgb_tfce = xgb.XGBRegressor()
    xgb_tfce.load_model(f'{path[:-18]}/ml_models/tfce_model.txt')

    nexp = exp_df.shape[0]
    if nexp > 150:
        print('Dataset features parameters that would lead to Out-Of-Distribution cutoff prediction prediction: Accuracy can\'t be guaranteed. Please disable cutoff prediction!')
        exit()
    nsub = exp_df.Subjects
    nfoci = exp_df.Peaks
    features = feature_extraction(nexp, nsub, nfoci)
    
    vfwe_cutoff = xgb_vfwe.predict(features)    
    cfwe_cutoff = np.round(xgb_cfwe.predict(features))
    tfce_cutoff = xgb_tfce.predict(features)

    return vfwe_cutoff, cfwe_cutoff, tfce_cutoff


