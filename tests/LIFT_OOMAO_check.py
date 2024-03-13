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
import json

# Local modules
from LIFT.modules.Telescope import Telescope
from LIFT.modules.Detector  import Detector
from LIFT.modules.Source    import Source
from LIFT.modules.Zernike   import Zernike
from LIFT.modules.LIFT      import LIFT

# Local auxillary modules
from LIFT.tools.misc import magnitudeFromPSF, TruePhotonsFromMag, draw_PSF_difference

with open('configs/path_info.json') as f:
    config = json.load(f)

data_dir = config['path_pupil']

with fits.open(data_dir+'pupil_largehex_4096_ROT008.fits') as hdul:
    pupil_big = hdul[0].data.astype('float')

#pupil_small = resize(pupil_big, (128, 128), anti_aliasing=False) # mean pooling, spiders are lost
#pupil_small = skimage.measure.block_reduce(pupil_small, (2,2), np.min) # min pooling to preserve spiders
#pupil_small[np.where(pupil_small>0.01)] = 1.0

with fits.open(data_dir+'keckPupil64x64pixForCrossComparison.fits') as hdul:
    pupil_small = hdul[0].data.astype('float')

with fits.open(data_dir+'refFrameLift0p5RadAst6_4pixPerFwhm10masPerPixel.fits') as hdul:
    PSF_check = hdul[0].data.astype('float')

#hdu = fits.PrimaryHDU(data=pupil_small.astype(np.float32))
#hdu.writeto(data_dir+'pupil_small.fits')

#%%  ============================ Code in this cell is must have ============================
FWHM_pix = 4
f_D_0 = 10.949

radian2mas = 206264806.71915
ang_pixel = 2.157e-6/10.949*radian2mas/FWHM_pix #[mas]
wavelength = 2.157e-6 #[m]
f_D = 1./(FWHM_pix * ang_pixel/radian2mas / wavelength)
c = 4.3796/4
D = 10.0 # [m]
f = f_D * D # [m]
pixel_size = f * ang_pixel/radian2mas * c # [m]
#zeroPaddingFactor = f / pixel_size * wavelength / D 


# Lets imagine we have DITs of 10 frames with tota 1 second exposure
# This is to generate a synthetic PSF 
sampling_time = 0.002 # [s]
num_samples = 1

# If the number of pixels in image is odd, the code will automatically center generated PSFs it to one pixel in the middle
tel = Telescope(img_resolution        = 40,
                    pupil             = pupil_small,
                    diameter          = D,
                    focalLength       = f,
                    pupilReflectivity = 0.6,
                    gpu_flag          = True)

det = Detector(pixel_size     = pixel_size,
                sampling_time = sampling_time,
                samples       = num_samples,
                RON           = 5.0, # used to generate PSF or the synthetic R_n [photons]
                QE            = 1.0) # not used
det.object = None
det * tel
ngs_poly = Source([(wavelength, 12.0)]) # Initialize a target
ngs_poly * tel # attach the source to the telescope

# Initialize modal basis
Z_basis = Zernike(modes_num=10)
Z_basis.computeZernike(tel, normalize_unit=True)

diversity_shift = -172e-9 #[m]
OPD_diversity = Z_basis.Mode(4)*diversity_shift

#%%  ============================ Code in this cell is NOT must have ============================
# Synthetic PSF
coefs_0 = np.array([0, -100, 200, 0, 0, 0, 0, 0, 0, 0])*1e-9 #[m]
#coefs_0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])*1e-9 #[m]

def PSFfromCoefs(coefs):
    tel.src.OPD = Z_basis.wavefrontFromModes(tel,coefs) + OPD_diversity
    PSF = tel.ComputePSF()
    tel.src.OPD *= 0.0 # zero out just in case
    return PSF

PSF = PSFfromCoefs(coefs_0)
PSF_noisy_DITs, _ = tel.det.getFrame(PSF, noise=True, integrate=False)
#R_n = PSF_noisy_DITs.var(axis=2)    # LIFT flux-weighting matrix
PSF_0 = cp.array(PSF_noisy_DITs.mean(axis=2)) # input PSF

PSF_check_1 = cp.array(PSF_check / PSF_check.max())
PSF_1 = PSF / PSF.max()

diff = PSF_check_1 - PSF_1
lims = cp.array([cp.abs(diff.min()), cp.abs(diff.max())]).max().get()

draw_PSF_difference(PSF_1.get(), PSF_check_1.get(), diff.get(), is_log=False, diff_clims=None, crop=None, colormap='pink')

#import numpy as np
#from astropy.io import fits
#
#hdu = fits.PrimaryHDU(data=PSF.get().astype(np.float32))
#hdu.writeto(data_dir+'py_PSF.fits')

#%%  ============================ Code in this cell is must have ============================
from LIFT.modules.LIFT import LIFT

estimator = LIFT(tel, Z_basis, OPD_diversity, 20)

modes = [0,1,2]
coefs_1, PSF_1, _ = estimator.Reconstruct(PSF_0, R_n='iterative', mode_ids=modes, optimize_norm='sum')
print(np.round(coefs_1*1e9,0))


#%%  ============================ Code in this cell is NOT must have ============================
'''
def GenerateWFE(coefs):
    if hasattr(Z_basis.modesFullRes, 'device'):
        Wf_aberrated = (Z_basis.modesFullRes[:,:,modes]*cp.array(coefs[modes])).sum(axis=2) # [m]
        return cp.asnumpy( (OPD_diversity + Wf_aberrated)*1e9 ) #[nm]
    else:
        Wf_aberrated = (Z_basis.modesFullRes[:,:,modes]*coefs[modes]).sum(axis=2) # [m]        
        return (OPD_diversity + Wf_aberrated)*1e9 #[nm]

WF_0 = GenerateWFE(coefs_0)
WF_1 = GenerateWFE(coefs_1)

plt.imshow(np.hstack([WF_0, WF_1, np.abs(WF_0-WF_1)]))
plt.show()

print('WFE: ', np.round(np.std(WF_0-WF_1)).astype('int'), '[nm]')

#plt.imshow(np.hstack([np.log(PSF_0), np.log(PSF_1), np.log(np.abs(PSF_0-PSF_1))]))
plt.imshow(np.hstack([PSF_0, PSF_1, np.abs(PSF_0-PSF_1)]))
plt.show()

print('Coefficients difference [nm]: ', *np.round((coefs_0[:5]-coefs_1)*1e9).astype('int').tolist() )
'''
#%% ------- Linearity range scanning -------
estimator = LIFT(tel, Z_basis, OPD_diversity, 50)

def_scan = np.arange(-1500, 1500+1, 50)*1e-9
modes = [0,1,2]

PSFs_0 = []
PSFs_1 = []

norm_regimes = {
    'no norm': None,
    'sum': 'sum',
    'max': 'max'
}

R_n = None
R_n_regimes = {
    'RON eye matrix': None,
    'image cube': R_n,
    'model': 'model',
    'iterative': 'iterative'
}

n_regime = 'sum'
R_regime = 'iterative'

Ns_ph = []

defocuses_est = []
for i in range(10):
    N_ph = []
    defocus_est = []
    for defocus in tqdm(def_scan):
        coefs_def = np.zeros(10)
        coefs_def[2] = defocus

        PSF_noisy_DITs, _ = tel.det.getFrame(PSFfromCoefs(coefs_def), noise=True, integrate=False)
        R_n = PSF_noisy_DITs.var(axis=2) 
        PSF_0 = PSF_noisy_DITs.mean(axis=2)
        #PSF_0 = PSF_noisy_DITs
        N_ph.append(PSF_0.sum())
        coefs_1, PSF_1, _ = estimator.Reconstruct(PSF_0, R_n=R_n_regimes[R_regime], mode_ids=modes, optimize_norm=norm_regimes[n_regime])
        PSFs_0.append(np.copy(PSF_0))
        PSFs_1.append(np.copy(PSF_1))
        defocus_est.append(coefs_1)
    defocuses_est.append( np.array(defocus_est) )
    Ns_ph.append(np.array(N_ph))
defocuses_est = np.array(defocuses_est)

errs = defocuses_est.std(axis=0)
vals = defocuses_est.mean(axis=0)
#%%
ax = plt.gca() #you first need to get the axis handle
ax.set_aspect(1)
plt.plot(np.array([def_scan.min()*1e9, def_scan.max()*1e9]), np.array([def_scan.min()*1e9, def_scan.max()*1e9]), '--',color='gray')
plt.errorbar(def_scan*1e9, vals[:,0]*1e9, yerr=errs[:,0]*1e9, label='Tip')
plt.errorbar(def_scan*1e9, vals[:,1]*1e9, yerr=errs[:,1]*1e9, label='Tilt')
plt.errorbar(def_scan*1e9, vals[:,2]*1e9, yerr=errs[:,2]*1e9, label='Defocus')
plt.legend()
plt.grid()
plt.xlim([def_scan.min()*1e9, def_scan.max()*1e9])
plt.ylim([def_scan.min()*1e9, def_scan.max()*1e9])
plt.xlabel('Defocus [nm]')
plt.ylabel('Defocus [nm]')
plt.title('TTF, RON='+str(det.readoutNoise)+', '+n_regime+', $R_n$='+R_regime)
#plt.show()

plt.savefig('data/plots/TTF_'+str(det.readoutNoise)+'_'+n_regime+'_'+R_regime+'.pdf')

#%%
'''
PSFs_0_ = np.dstack(PSFs_0)
PSFs_1_ = np.dstack(PSFs_1)
comparison = np.hstack([PSFs_0_, PSFs_1_])


def save_GIF(array, duration=1e3, scale=1, path='test.gif'):
    from PIL import Image
    from matplotlib import cm
    from skimage.transform import rescale
    
    gif_anim = []
    for layer in np.rollaxis(array, 2):
        buf = layer/layer.max()
        if scale != 1.0:
            buf = rescale(buf, scale, order=0)
        gif_anim.append( Image.fromarray(np.uint8(cm.viridis(buf)*255)) )
    gif_anim[0].save(path, save_all=True, append_images=gif_anim[1:], optimize=True, duration=duration, loop=0)


def save_GIF_RGB(images_stack, duration=1e3, downscale=4, path='test.gif'):
    from PIL import Image
    gif_anim = []
    
    def remove_transparency(im, bg_colour=(255, 255, 255)):
        alpha = im.convert('RGBA').split()[-1]
        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg
    
    for layer in images_stack:
        im = Image.fromarray(np.uint8(layer*255))
        im.thumbnail((im.size[0]//downscale, im.size[1]//downscale), Image.ANTIALIAS)
        gif_anim.append( remove_transparency(im) )
        gif_anim[0].save(path, save_all=True, append_images=gif_anim[1:], optimize=True, duration=duration, loop=0)


save_GIF(comparison, duration=1e3, scale=4, path='test.gif')

plt.imshow(comparison[:,:,-1])
id = 1

coefs_def = np.zeros(10)
coefs_def[2] = vals[id,2]
test1, _ = tel.det.getFrame(PSFfromCoefs(coefs_def), noise=False, integrate=True)
coefs_def[2] = def_scan[id]
test0, _ = tel.det.getFrame(PSFfromCoefs(coefs_def), noise=False, integrate=True)

plt.imshow( np.hstack([test0, test1, PSFs_0_[:,:,id], PSFs_1_[:,:,id]]) )
'''
#%%
