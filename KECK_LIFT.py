#%%
%reload_ext autoreload
%autoreload 2
# Commom modules
import matplotlib.pyplot as plt
import numpy as np

try:
    import cupy as cp
except ImportError or ModuleNotFoundError:
    print('CuPy is not found, using NumPy backend...')
    cp = np

from astropy.io import fits
from skimage.transform import resize
import skimage.measure
from tqdm import tqdm

# Local modules
from modules.Telescope import Telescope
from modules.Detector  import Detector
from modules.Source    import Source
from modules.Zernike   import Zernike
from modules.LIFT      import LIFT

# Local auxillary modules
from tools.misc import draw_PSF_difference

#with fits.open('C:\\Users\\akuznets\\Data\\KECK\\pupil_largehex_4096_ROT008.fits') as hdul:
#    pupil_big = hdul[0].data.astype('float')
#pupil_small = resize(pupil_big, (64, 64), anti_aliasing=False) # mean pooling, spiders are lost
#pupil_small[np.where(pupil_small<1.0)] = 0.0
##pupil_small = skimage.measure.block_reduce(pupil_big, (64,64), np.min) # min pooling to preserve spiders

with fits.open('C:\\Users\\akuznets\\Data\\KECK\\keckPupil64x64pixForCrossComparison.fits') as hdul:
    pupil_small = hdul[0].data.astype('float')

#plt.imshow(pupil_small)

#%%  ============================ Code in this cell is must have ============================
D = 10.0 # [m]
pixel_size = 18e-6 # [m]
ang_pixel = 50 # [mas]
f = pixel_size / ang_pixel * 206264806.71915 # [m]

# Lets imagine we have DITs of 10 frames with tota 1 second exposure
# This is to generate a synthetic PSF 
sampling_time = 0.1 # [s]
num_samples = 10

# If the number of pixels in image is odd, the code will automatically center generated PSFs it to one pixel in the middle
tel = Telescope(img_resolution        = 17,
                    pupil             = pupil_small,
                    diameter          = D,
                    focalLength       = f,
                    pupilReflectivity = 1.0,
                    gpu_flag          = True)

det = Detector(pixel_size     = pixel_size,
                sampling_time = sampling_time,
                samples       = num_samples,
                RON           = 5.0, # used to generate PSF or the synthetic R_n [photons]
                QE            = 0.7) # not used

det.object = None
det * tel
ngs_poly = Source([('H', 19.0)]) # Initialize a target of H_mag=10
ngs_poly * tel # attach the source to the telescope

# Initialize modal basis
Z_basis = Zernike(modes_num = 10)
Z_basis.computeZernike(tel)

diversity_shift = 200e-9 #[m]
OPD_diversity = Z_basis.Mode(3)*diversity_shift

#%%  ============================ Code in this cell is NOT must have ============================
# Synthetic PSF
coefs_0 = np.array([10, -150, 200, 20, -45, 34, 51, -29, 20, 10])*1e-9 #[m]
#coefs_0 = np.array([0, 0, 200, 0, 0, 0, 0, 0, 0, 0])*1e-9 #[m]

def PSFfromCoefs(coefs):
    if hasattr(Z_basis.modesFullRes, 'device'):
        wavefront = Z_basis.modesFullRes.get() @ coefs + OPD_diversity.get() * tel.pupil.get()
    else:
        wavefront = Z_basis.modesFullRes @ coefs + OPD_diversity * tel.pupil
        
    tel.src.OPD = wavefront
    PSF = tel.ComputePSF()
    tel.src.OPD *= 0.0 # zero out just in case
    return PSF

PSF = PSFfromCoefs(coefs_0)
PSF_noisy_DITs, _ = tel.det.getFrame(PSF, noise=True, integrate=False)
R_n = PSF_noisy_DITs.var(axis=2)    # LIFT flux-weighting matrix
PSF_0 = PSF_noisy_DITs.mean(axis=2) # input PSF


#%%  ============================ Code in this cell is must have ============================
estimator = LIFT(tel, Z_basis, OPD_diversity, 20)

# Increase the flux artificially to test the normalization optimization inside the LIFT
PSF_0 *= 4
R_n *= 4

#modes = [0,1,2,3,4,5,6,7,8,9]
modes = [0,1,2,3,4]
#           Flux optimization is something to be reconsidered --------------V
#coefs_1, PSF_1, _ = estimator.Reconstruct(PSF_0, R_n=R_n, mode_ids=modes, optimize_norm='sum')
coefs_1, PSF_1, _ = estimator.Reconstruct(PSF_0, R_n='model', mode_ids=modes, optimize_norm='sum')

#%  ============================ Code in this cell is NOT must have ============================
def GenerateWFE(coefs):
    if hasattr(Z_basis.modesFullRes, 'device'):
        WF_aberrated = (Z_basis.modesFullRes[:,:,modes]*cp.array(coefs[modes])).sum(axis=2) # [m]
        return cp.asnumpy( (OPD_diversity + WF_aberrated)*1e9 ) #[nm]
    else:
        WF_aberrated = (Z_basis.modesFullRes[:,:,modes]*coefs[modes]).sum(axis=2) # [m]        
        return (OPD_diversity + WF_aberrated)*1e9 #[nm]

WF_0 = GenerateWFE(coefs_0)
WF_1 = GenerateWFE(coefs_1)

plt.imshow(np.hstack([WF_0, WF_1, np.abs(WF_0-WF_1)]))
plt.show()

print('WFE: ', np.round(np.std(WF_0-WF_1)).astype('int'), '[nm]')

#plt.imshow(np.hstack([np.log(PSF_0), np.log(PSF_1), np.log(np.abs(PSF_0-PSF_1))]))
plt.imshow(np.hstack([PSF_0, PSF_1, np.abs(PSF_0-PSF_1)]))
plt.show()

print('Coefficients difference [nm]: ', *np.round((coefs_0[:5]-coefs_1)*1e9).astype('int').tolist() )

#%% ------- Linearity range scanning -------
def_scan = np.arange(-100,101,10)*1e-9
modes = [0,1,2,3,4]

defocus_est = []

for defocus in tqdm(def_scan):
    coefs_def = np.zeros(10)
    coefs_def[2] = defocus

    PSF_noisy_DITs, _ = tel.det.getFrame(PSFfromCoefs(coefs_def), noise=True, integrate=False)
    R_n = PSF_noisy_DITs.var(axis=2) 
    PSF_0 = PSF_noisy_DITs.mean(axis=2)

    coefs_1, PSF_1, _ = estimator.Reconstruct(PSF_0, R_n=R_n, mode_ids=modes, optimize_norm=False)
    defocus_est.append(coefs_1[2])

defocus_est = np.array(defocus_est)
ax = plt.gca() #you first need to get the axis handle
ax.set_aspect(1)
plt.plot(def_scan*1e9, defocus_est*1e9)
plt.plot(np.array([def_scan.min()*1e9, def_scan.max()*1e9]), np.array([def_scan.min()*1e9, def_scan.max()*1e9]))
plt.grid()
plt.xlim([def_scan.min()*1e9, def_scan.max()*1e9])
plt.ylim([def_scan.min()*1e9, def_scan.max()*1e9])
plt.xlabel('Defocus [nm]')
plt.ylabel('Defocus [nm]')
plt.title('Linearity range scan ('+str(ang_pixel)+' mas)')
plt.show()
# %%
