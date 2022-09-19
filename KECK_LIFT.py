#%%
# Commom modules
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from skimage.transform import resize
import skimage.measure

# Local modules
from modules.Telescope import Telescope
from modules.Detector  import Detector
from modules.Source    import Source
from modules.Zernike   import Zernike
from modules.LIFT      import LIFT

# Local auxillary modules
from tools.misc import magnitudeFromPSF, TruePhotonsFromMag

with fits.open('C:\\Users\\akuznets\\Data\\KECK\\pupil_largehex_4096_ROT008.fits') as hdul:
    pupil_big = hdul[0].data.astype('float')

pupil_small = resize(pupil_big, (64, 64), anti_aliasing=False) # mean pooling, spiders are lost
#pupil_small = skimage.measure.block_reduce(pupil_big, (64,64), np.min) # min pooling to preserve spiders


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
tel = Telescope(img_resolution    = 33,
                    pupil             = pupil_small,
                    diameter          = D,
                    focalLength       = f,
                    pupilReflectivity = 1.0)

det = Detector(pixel_size = pixel_size,
                sampling_time = sampling_time,
                samples       = num_samples,
                RON           = 4.0, #used to generate PSF or the synthetic R_n
                QE            = 1.0) #not used
det.object = None
det * tel
ngs_poly = Source([('H', 10.0)]) # Initialize a target of H_mag=10
ngs_poly * tel # attach the source to the telescope

# Initialize modal basis
Z_basis = Zernike(modes_num = 10)
Z_basis.computeZernike(tel)

diversity_shift = 200e-9 #[m]
OPD_diversity = Z_basis.Mode(3)*diversity_shift


#%%  ============================ Code in this cell is NOT must have ============================
# Synthetic PSF
coefs_0 = np.array([10, -15, 200, 20, -45, 34, 21, -29, 20, 10])*1e-9 #[m]

def PSFfromCoefs(coefs):
    wavefront = Z_basis.modesFullRes @ coefs + OPD_diversity
    tel.src.OPD = wavefront
    PSF = tel.ComputePSF()
    tel.src.OPD *= 0.0 # zero out just in case
    return PSF

PSF = PSFfromCoefs(coefs_0)
PSF_noisy_DITs, _ = tel.det.getFrame(PSF, noise=True, integrate=False)

# v-------- Or initialize it synthetically. You know better :)
R_n = PSF_noisy_DITs.var(axis=2)    # LIFT flux-weighting matrix
PSF_0 = PSF_noisy_DITs.mean(axis=2) # input PSF

R_n =  R_n * 0 + 1.0 #okay, lets assume it's just all ones for now

#%%  ============================ Code in this cell is must have ============================
estimator = LIFT(tel, Z_basis, OPD_diversity, 20)

#modes = [0,1,2,3,4,5,6,7,8,9]
modes = [0,1,2,3,4]
#           Flux optimization is something to be reconsidered --------------V
coefs_1, PSF_1, _ = estimator.Reconstruct(PSF_0, R_n=R_n, mode_ids=modes, optimize_norm=False)

#%%  ============================ Code in this cell is NOT must have ============================
def GenerateWFE(coefs):
    Wf_aberrated = np.sum(Z_basis.modesFullRes[:,:,modes]*coefs[modes], axis=2) # [m]
    return (OPD_diversity + Wf_aberrated)*1e9 #[nm]

WF_0 = GenerateWFE(coefs_0)
WF_1 = GenerateWFE(coefs_1)

plt.imshow(np.hstack([WF_0, WF_1, np.abs(WF_0-WF_1)]))
plt.show()

print('WFE: ', np.round(np.std(WF_0-WF_1)).astype('int'), '[nm]')

plt.imshow(np.hstack([np.log(PSF_0), np.log(PSF_1), np.log(np.abs(PSF_0-PSF_1))]))
plt.show()

# %%
