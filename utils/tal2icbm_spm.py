import numpy as np

def tal2icbm_spm(inpoints):
    
    '''This function converts coordinates from Talairach space to MNI
    space (normalized using the SPM software package) using the 
    tal2icbm transform developed and validated by Jack Lancaster 
    at the Research Imaging Center in San Antonio, Texas.'''
    
    icbm_spm = np.array(([0.9254, 0.0024, -0.0118, -1.0207],
                        [-0.0048, 0.9316, -0.0871, -1.7667],
                        [0.0152, 0.0883,  0.8924,  4.0926],
                        [0.0000, 0.0000,  0.0000,  1.0000]))

    icbm_spm = np.linalg.inv(icbm_spm)
    inpoints = np.pad(inpoints, ((0,0),(0,1)), 'constant', constant_values=1)
    inpoints = np.dot(icbm_spm, inpoints.T)
    outpoints = np.round(inpoints[:3])

    return outpoints.T