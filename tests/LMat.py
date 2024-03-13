#%%
# Commom modules
import numpy as np
from modules.Telescope import Telescope
from modules.Detector  import Detector
from modules.Source    import Source
from modules.Zernike   import Zernike
from modules.LIFT      import LIFT
from astropy.io import fits
import json

with open('configs/path_info.json') as f:
    config = json.load(f)

data_dir = config['path_pupil']

with fits.open(data_dir+'keckPupil64x64pixForCrossComparison.fits') as hdul:
    pupil_big = hdul[0].data.astype('float')


test_file = data_dir+'LIFT/LIFT/20200108lift_zc4_8x.fits'

with fits.open(test_file) as hdul: datacube = hdul[0].data
PSFs = datacube.mean(axis=1)

results = []

#%%
for i in range(PSFs.shape[0]):
    PSF_0           = np.asarray(PSFs[i,:,:])
    pupil_big       = np.asarray(pupil_big)
    dec_res         = 8
    R_n             = np.ones_like(PSF_0)*25
    diversity_shift = np.asarray(-253e-9)
    d_modp          = 4
    modes           = np.asarray([0,1,2]).tolist()
    ang_pixel       = np.asarray(50)
    D               = np.asarray(10.0)
    NGS_Band        = 'Ks'
    sampling_time   = np.asarray(5e-3)
    
    QE = 0.7
    RON = 5
    Hmag = 10.0

    pupil_small = pupil_big

    pixel_size = 18e-6 # [m]
    f = pixel_size / ang_pixel * 206264806.71915 # [m]

    tel = Telescope(img_resolution    = dec_res,
                    pupil             = pupil_small,
                    diameter          = D,
                    focalLength       = f,
                    pupilReflectivity = 1.0)

    det = Detector(pixel_size     = pixel_size,
                sampling_time = sampling_time,
                samples       = 1,
                RON           = RON, #used to generate PSF or the synthetic R_n
                QE            = QE) #not used
    det.object = None
    det * tel

    ngs_poly = Source([(NGS_Band, Hmag)]) # Initialize a target of H_mag=10
    ngs_poly * tel # attach the source to the telescope

    print(type(NGS_Band))

    # Initialize modal basis
    Z_basis = Zernike(modes_num = 10)
    Z_basis.computeZernike(tel)

    OPD_diversity = Z_basis.Mode(d_modp)*diversity_shift
    estimator = LIFT(tel, Z_basis, OPD_diversity, 20)
    coefs_1, PSF_1, _ = estimator.Reconstruct(PSF_0, R_n=R_n, mode_ids=modes, optimize_norm='sum')

    results.append(coefs_1)

# %%
import matplotlib.pyplot as plt

A_caps = np.array(results)

x = np.arange(A_caps.shape[0], dtype='int')
for i in np.arange(A_caps.shape[1]):
    plt.plot(x,A_caps[:,i]*1e9, label=Z_basis.modeName(i))
plt.ylim([-500,500])
plt.grid()
plt.axhline(linestyle='--', c='gray')
plt.axvline(15, linestyle='--', c='gray')
plt.legend()

# %%
