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
from tools.misc import draw_PSF_difference, mask_circle
from tools.fit_gaussian import fitgaussian, plot_gaussian

#%%  ============================ Code in this cell is must have ============================
with fits.open('C:\\Users\\akuznets\\Data\\KECK\\keckPupil64x64pixForCrossComparison.fits') as hdul:
    pupil_small = hdul[0].data.astype('float')

#pupil_small = mask_circle(pupil_small.shape[0], pupil_small.shape[0]//2)

#D = 11.732 # [m]
D = 10 # [m]
wavelength = 2.157e-6
pixel_size = 18e-6 # [m]
ang_pixel = 50 # [mas]
f = pixel_size / ang_pixel * 206264806.71915 # [m]

# Lets imagine we have DITs of 10 frames with tota 1 second exposure
# This is to generate a synthetic PSF 
sampling_time = 0.1 # [s]
num_samples = 30

# If the number of pixels in image is odd, the code will automatically center generated PSFs it to one pixel in the middle
tel = Telescope(img_resolution        = 8,
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
ngs = Source([(wavelength, 10.0)]) # Initialize a target of H_mag=10
ngs * tel # attach the source to the telescope

# Initialize modal basis
Z_basis = Zernike(modes_num = 10)
Z_basis.computeZernike(tel)

#diversity_shift = -172e-9 #[m]
diversity_shift = -253e-9 #[m]
OPD_diversity = Z_basis.Mode(4)*diversity_shift

#%%
test_file = 'C:\\Users\\akuznets\\Data\\KECK\\LIFT\\LIFT\\20200108lift_zc4_8x.fits'

with fits.open(test_file) as hdul: datacube = hdul[0].data
PSFs = datacube.mean(axis=1)

#%%  ============================ Code in this cell is NOT must have ============================
# Synthetic PSF
coefs_0 = np.array([-500, 200, -200, 20, -45, 34, 51, -29, 20, 10])*1e-9 #[m]
#coefs_0 = np.array([0, 0, 200, 0, 0, 0, 0, 0, 0, 0])*1e-9 #[m]
#coefs_0 = np.array([0, 0, 200, 0, 0, 0, 0, 0, 0, 0])*1e-9 #[m]

def PSFfromCoefs(coefs):
    tel.src.OPD = Z_basis.wavefrontFromModes(tel,coefs) + OPD_diversity
    PSF = tel.ComputePSF()
    tel.src.OPD *= 0.0 # zero out just in case
    return PSF

PSF = PSFfromCoefs(coefs_0)
PSF_noisy_DITs, _ = tel.det.getFrame(PSF, noise=True, integrate=False)
R_n = PSF_noisy_DITs.var(axis=2)    # LIFT flux-weighting matrix
PSF_0 = PSF_noisy_DITs.mean(axis=2) # input PSF

def IntialTT(im):
    N = im.shape[0]
    factor = 0.5 * (1-N%2)
    coord_range = np.linspace(-N//2+N%2+factor, N//2-factor, N)
    xx, yy = np.meshgrid(coord_range, coord_range)

    w_avg = lambda x, w: (x*w).sum()/w.sum()

    def threshold(x, t):
        x_ = np.copy(x)
        x_[np.where(x_ < x_.max()*t)] = 0.0
        return x_

    x = w_avg(xx, threshold(im, t=0.25))
    y = w_avg(yy, threshold(im, t=0.25))

    Z_0_max = (Z_basis.Mode(0)).max().item()
    Z_1_max = (Z_basis.Mode(1)).max().item()

    tilt = D*pixel_size*x/2/f/Z_1_max
    tip  = D*pixel_size*y/2/f/Z_0_max
    return tip, tilt


def IntialDefocus(im):
    p = fitgaussian(im)
    return np.clip(np.sign(diversity_shift) * 250e-9*(p[3]/p[4]-1), a_min=-200-9, a_max=200e-9)


#print(IntialTT(PSF_0)[0]*1e9, IntialTT(PSF_0)[1]*1e9, IntialDefocus(PSF_0)*1e9)
#plt.imshow(PSF_0)
#plt.show()

#%%  ============================ Code in this cell is must have ============================
estimator = LIFT(tel, Z_basis, OPD_diversity, 20)

# Increase the flux artificially to test the robustness of the normalization
# optimization inside the LIFT

A_caps = []

for i in range(21):
    PSF_0 = PSFs[i,:,:]
    modes = [0,1,2,3]

    #tip, tilt = IntialTT(PSF_0)
    #defocus = IntialDefocus(PSF_0)
    #A_init = None
    #A_init = np.zeros(max(modes)+1)
    #A_init[0:3] = (tip*0.5, tilt*0.5, defocus*0.5)

    coefs_1, PSF_1, _ = estimator.Reconstruct(PSF_0, R_n=None, mode_ids=modes, optimize_norm='sum')

    A_caps.append(coefs_1)

    to_show_PSF = np.hstack([PSF_0, PSF_1, np.abs(PSF_0-PSF_1)])
    #plt.imshow(to_show_PSF)
    #plt.show()

A_caps = np.array(A_caps)


x = np.arange(A_caps.shape[0], dtype='int')
for i in np.arange(A_caps.shape[1]):
    plt.plot(x,A_caps[:,i]*1e9, label=Z_basis.modeName(i))
plt.ylim([-500,500])
plt.grid()
plt.axhline(linestyle='--', c='gray')
plt.axvline(15, linestyle='--', c='gray')
plt.legend()

# %%
