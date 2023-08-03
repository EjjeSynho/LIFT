#%%
import sys
sys.path.append("..")

from astropy.io import fits
from scipy.io import readsav
from tqdm import tqdm
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

from modules.Telescope import Telescope
from modules.Detector  import Detector
from modules.Source    import Source
from modules.Zernike   import Zernike
from modules.LIFT      import LIFT

from tools.misc import mask_circle
from tools.fit_gaussian import fitgaussian, gaussian

# Initialize data directories
root_K = 'C:/Users/akuznets/Data/KECK/LIFT/Mojtaba_new/20221218_K/'
root_H = 'C:/Users/akuznets/Data/KECK/LIFT/Mojtaba_new/20230201_H/'

files = [
    root_K + 'TRICK_K11.2_cpr78_coadd3_bgd4nd3.sav',
    root_K + 'TRICK_K11.2_cpr78_coadd3_nd3.sav',
    root_K + 'TRICK_K12.3_cpr309_coadd3_bgd4nd5.sav',
    root_K + 'TRICK_K12.3_cpr309_coadd3_nd5.sav',
    root_K + 'TRICK_K13.8_cpr2001_coadd3_bgd4nd6.sav',
    root_K + 'TRICK_K13.8_cpr2001_coadd3_nd6.sav', 
    root_H + 'TRICK_Hmag11.0_R13.2_cpr20_coadds1_622Hz.sav',
    root_H + 'TRICK_Hmag11.0_R13.2_cpr20_coadds1_622Hz_bgd.sav',
    root_H + 'TRICK_Hmag13.1_R15.5.sav',
    root_H + 'TRICK_Hmag13.1_R15.5_bgd.sav',
    root_H + 'TRICK_Hmag8.6_R11.9_cpr9_coadds1_622Hz.sav',
    root_H + 'TRICK_Hmag8.6_R11.9_cpr9_coadds1_622Hz_bgd.sav'
]

with fits.open('C:/Users/akuznets/Data/KECK/circular_pupil_64.fits') as hdul:
    pupil_big = hdul[0].data.astype('float')

selection = files[-2]

data    = readsav(selection)
cube    = data['frame_cube']
defocus = data['defocus']


cube = cube[:, :, 1:-1, 1:-1] - np.median(cube) # remove backround
cube /= np.sum(cube, axis=(-2,-1), keepdims=True) # Normalize the cube

ang_pixel     = 50.0
sampling_time = 5e-3
pixel_size    = 18e-6 # [m]

selection = selection.split('/')[-1].split('.sav')[0] # separate the name of the dataset
band = selection[6] # select the letter K or H

# %%
# Initialize the image cropper around the center
crop = 13
size = cube.shape[-1]
ROI = slice(size//2-crop//2-crop%2, size//2+crop//2)
ROI = (ROI, ROI)

keck = Telescope(img_resolution    = crop,
                 pupil             = mask_circle(64, 32, center=(0,0), centered=True), # a circular pupil is assumed
                 diameter          = 11.732,
                 focalLength       = pixel_size / ang_pixel * 206264806.71915,
                 pupilReflectivity = 1.0,
                 gpu_flag          = True)

det = Detector(pixel_size    = pixel_size,
               sampling_time = sampling_time,
               samples       = 1,
               RON           = 4.0,
               QE            = 0.7)

ngs = Source([(band, 0.0)])
det * keck
ngs * keck

# Initialize dataset-dependent parameters
if band == 'K':
    sigma_obj = 0.65
    modes = [0,1,2,3] # K PSFs seem to be slightly rotated, so we need to add astigmatism
    def_prev = lambda sigma_x, sigma_y: (sigma_x/sigma_y-1+0.1) * 750 * 0.8# Handcrafted expression for defocus from PSF elongation
    
if band == 'H':
    sigma_obj = 0.4
    modes = [0,1,2]
    def_prev = lambda sigma_x, sigma_y: (sigma_x/sigma_y-1+0.1) * 750 * 1.1

# Generate the convolution kernel
obj_resolution = 16
xx, yy = np.meshgrid(np.arange(0, obj_resolution), np.arange(0, obj_resolution))
obj1 = np.abs( gaussian(1.0, obj_resolution/2-1, obj_resolution/2-1,  sigma_obj, sigma_obj)(xx,yy) )
obj1 = obj1[:-1,:-1]
keck.object = cp.array(obj1) if keck.gpu else obj1 # Important! Initializethe convolutiOn kernel

# Initialize modal basis
Z = Zernike(modes_num=50)
Z.computeZernike(keck)

diversity_shift = 200e-9    
new_diversity = Z.Mode(4)*diversity_shift

zero_defocus_id = np.where(defocus == 0.0)[0][0]

estimator = LIFT(keck, Z, new_diversity, 20)

PSF_0 = cube[zero_defocus_id,:,...].mean(axis=0)[ROI]
R_n   = cube[zero_defocus_id,:,...].var(axis=0)[ROI]


#%% Diverity calibration to determine 
A_1, PSF_1, _ = estimator.Reconstruct(PSF_0, R_n=R_n, mode_ids=modes, optimize_norm='sum')
plt.imshow(np.hstack([PSF_0, PSF_1]))

print("Expected defocus: {:.1f} [nm]\nMeasured defocus {:.1f} [nm]".format(defocus[zero_defocus_id], (A_1[2])*1e9))

A_1[0:2] *= 0 # Remove TT component, if you want to
new_diversity = Z.wavefrontFromModes(estimator, A_1) + Z.Mode(4)*diversity_shift # now diversity phase contains defocus
estimator.diversity_OPD = new_diversity

#%% Defocus ramp estimation
single_curve = True # if True, performs LIFT estimation for an averaged PSF

if single_curve:
    As = []
    As_init = []
    for i in tqdm(range(cube.shape[0])):
        A_ = []
        A_init_ = []
        R_n   = cube[i,:,...].var(axis=0)[ROI]
        PSF_0 = cube[i,:,...].mean(axis=0)[ROI]
        A_1_init = cp.array( [0, 0, def_prev(*fitgaussian(PSF_0)[-2:]) * 1e-9] ) # This coefficient vector is obtained from fitting 2d Gaussian to the PSF
        A_1, _, _ = estimator.Reconstruct(PSF_0, R_n=R_n, mode_ids=modes, A_0=A_1_init, optimize_norm='sum')
        
        A_.append(np.copy(A_1))
        A_init_.append(np.copy(A_1_init))
        
        As.append(np.stack(A_))
        As_init.append(np.stack(A_init_))

    As = np.stack(As)
    As_init = np.stack(As_init).get()

else:
    As = []
    As_init = []
    for i in tqdm(range(cube.shape[0])):
        A_ = []
        A_init_ = []
        R_n   = cube[i,:,...].var(axis=0)[ROI]
        for j in range(cube.shape[1]):
            PSF_0 = cube[i,j,...][ROI]
            A_1_init = cp.array( [0, 0, def_prev(*fitgaussian(PSF_0)[-2:]) * 1e-9] )
            A_1, _, _ = estimator.Reconstruct(PSF_0, R_n=R_n, mode_ids=modes, A_0=A_1_init, optimize_norm='sum')
            
            A_.append(np.copy(A_1))
            A_init_.append(np.copy(A_1_init))
            
        As.append(np.stack(A_))
        As_init.append(np.stack(A_init_))

    As = np.stack(As)
    As_init = np.stack(As_init).get()


#%%
mode_id = 2

defus = As.mean(axis=1)[:,mode_id] * 1e9
elongus = As_init.mean(axis=1)[:, mode_id] * 1e9

bias_LIFT  = 0 * defus[15]
bias_elong = elongus[15]

p = 95 # Initialize 95% confidence interval for LIFT and elongation estimates
d_upper = np.percentile(As[..., mode_id]*1e9, 100.0-p, axis=1)
d_lower = np.percentile(As[..., mode_id]*1e9, p,       axis=1)

e_upper = np.percentile(As_init[..., mode_id]*1e9, 100.0-p, axis=1)
e_lower = np.percentile(As_init[..., mode_id]*1e9, p,       axis=1)

plt.fill_between(defocus, d_lower-bias_LIFT,  d_upper-bias_LIFT,  color='tab:green', alpha=0.25)
plt.fill_between(defocus, e_lower-bias_elong, e_upper-bias_elong, color='tab:blue',  alpha=0.25)

plt.plot(defocus, defocus, color='gray', linestyle='--')
plt.plot(defocus, defus-bias_LIFT,   color='tab:green', label='LIFT')
plt.plot(defocus, elongus-bias_elong, color='tab:blue', label='Elongation')
plt.legend()
plt.grid()
ax = plt.gca()
ax.set_aspect('equal')
plt.xlim([defocus.min(), defocus.max()])
plt.ylim([defocus.min(), defocus.max()])
plt.xlabel('Defocus introduced, [nm]')
plt.ylabel('Defocus reconstructeded, [nm]')
plt.title(selection)

plt.savefig(selection.split('.sav')[0]+'_defocus.png', dpi=300, bbox_inches='tight')


# %%


