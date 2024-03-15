#%%
import sys
sys.path.append(".")
sys.path.append("..")

from matplotlib import pyplot as plt
import numpy as np
from tools.fit_gaussian import gaussian

import numpy as np
try:
    import cupy as cp
except ImportError or ModuleNotFoundError:
    print('CuPy is not installed, using NumPy backend...')
    cp = np


rad2mas  = 3600 * 180 * 1000 / np.pi
rad2arc  = rad2mas / 1000
deg2rad  = np.pi / 180
asec2rad = np.pi / 180 / 3600

seeing = lambda r0, lmbd: rad2arc*0.976*lmbd/r0 # [arcs]
r0_new = lambda r0, lmbd, lmbd0: r0*(lmbd/lmbd0)**1.2 # [m]
r0     = lambda seeing, lmbd: rad2arc*0.976*lmbd/seeing # [m]

Nph_diff = lambda m_1, m_2: 10**(-(m_2-m_1)/2.5)
mag_diff = lambda ph_diff: -2.5*np.log10(ph_diff)


def magnitudeFromPSF(tel, photons, band, sampling_time=None):
    xp = cp if tel.gpu else np
    if sampling_time is None:
        sampling_time = tel.det.sampling_time
    zero_point = tel.src.PhotometricParameters(band)[2]
    fluxMap = photons / tel.pupil.sum() * tel.pupil
    nPhoton = xp.nansum(fluxMap / tel.pupilReflectivity, axis=(-2,-1)) / (xp.pi*(tel.D/2)**2) / sampling_time
    return -2.5 * xp.log10(368 * nPhoton / zero_point )


def TruePhotonsFromMag(tel, mag, band, sampling_time): # [photons/aperture] !not per m2!
    xp = cp if tel.gpu else np
    c = tel.pupilReflectivity * xp.pi*(tel.D/2)**2*sampling_time
    return tel.src.PhotometricParameters(band)[2]/368 * 10**(-mag/2.5) * c


def optimal_img_size(tel, N_modes, force_odd=True, force_even=False):
    def zernIndex(j):
        n = int((-1.0 + np.sqrt(8*(j-1)+1))/2.)
        p = (j-(n*(n+1))/2.)
        k = n % 2
        m = int((p+k)/2.)*2 - k

        if m != 0:
            if j % 2 == 0:
                s = 1
            else:
                s = -1      
            m *= s
        return [n, m]
    
    n_radial = zernIndex(N_modes)[0] + 1

    n_pass = n_radial
    n_filt = n_radial + 2 + 1

    N_λ_D_1_pass, N_λ_D_1_filt = ( 0.6*(n_pass+1) + 1, 0.6*(n_filt+1) - 1 )

    N_pix = lambda N_λ_D, λ: int(np.ceil(2 * N_λ_D * λ/tel.D * tel.f/tel.det.pixel_size))
    
    N_pix_max = np.floor( N_pix(N_λ_D_1_filt, tel.src.spectrum[0]['wavelength']) )
    N_pix_min = np.ceil ( N_pix(N_λ_D_1_pass, tel.src.spectrum[0]['wavelength']) )

    N_pix = np.maximum(N_pix_min, N_pix_max).astype(np.uint16)
    
    if force_odd and N_pix % 2 == 0:
        N_pix += 1 - N_pix % 2
    else:
        if force_even and N_pix % 2 == 1:
            N_pix += N_pix % 2
        
    return N_pix


def decompose_wavefront(WF, modal_basis, pupil_mask):
    if WF.ndim == 2:
        WF = WF[..., None]
    # Ensure that the wavefronts and modal basis are masked according to the pupil mask
    WF_masked = WF * pupil_mask[:, :, None]
    modal_basis_masked = modal_basis * pupil_mask[:, :, None]
    
    # Flatten the valid pixels of the masked wavefronts and modal basis
    valid_pixels = pupil_mask.flatten() > 1e-12
    WF_2D_valid  = WF_masked.reshape((-1, WF.shape[-1]))[valid_pixels, :]
    modal_basis_2D_valid = modal_basis_masked.reshape((-1, modal_basis.shape[-1]))[valid_pixels, :]
    
    # Compute the projection using matrix multiplication, assuming cp.array is desired for GPU computation
    coefs = WF_2D_valid.T @ modal_basis_2D_valid / pupil_mask.sum()
    return coefs


modes_colors = [
    'tab:blue',
    'tab:orange',
    'tab:green',
    'tab:red',
    'tab:cyan',
    'tab:purple',
    'tab:brown',
    'tab:pink',
    'tab:gray',
    'tab:olive',
    'indigo',
    'mediumorchid',
    'darkorchid',
    'plum',
    'lightsteelblue',
    'cornflowerblue',
    'royalblue',
    'navy',
    'khaki',
    'gold',
    'goldenrod',
    'darkgoldenrod'
]


def mask_circle(N, r, center=(0,0), centered=True):
    factor = 0.5 * (1-N%2)
    if centered:
        coord_range = np.linspace(-N//2+N%2+factor, N//2-factor, N)
    else:
        coord_range = np.linspace(0, N-1, N)
    xx, yy = np.meshgrid(coord_range-center[1], coord_range-center[0])
    pupil_round = np.zeros([N, N], dtype=np.int32)
    pupil_round[np.sqrt(yy**2+xx**2) < r] = 1
    return pupil_round


# To make this function work, one must ensure that size of inp must be the multiple of N
def binning(inp, N, regime='sum'):
    if N == 1:
        return inp
    
    xp  = cp.get_array_module(inp)
    out = xp.stack(xp.split(xp.stack(xp.split(xp.atleast_3d(inp), inp.shape[0]//N, axis=0)), inp.shape[1]//N, axis=2))
    
    if    regime == 'max':  func = xp.max
    elif  regime == 'min':  func = xp.min
    elif  regime == 'mean': func = xp.mean
    else: func = xp.sum
    
    return xp.squeeze( xp.transpose( func(out, axis=(2,3), keepdims=True), axes=(1,0,2,3,4)) )


def GenerateObject(sigma_obj=None):
    obj_resolution = 16
    xx, yy = np.meshgrid(np.arange(0, obj_resolution), np.arange(0, obj_resolution))
    if sigma_obj is None:
        return None    
    elif hasattr(sigma_obj, '__len__'):
        obj1 = np.abs( gaussian(1.0, obj_resolution/2-1, obj_resolution/2-1,  sigma_obj[0], sigma_obj[1])(xx,yy) )
    else:
        obj1 = np.abs( gaussian(1.0, obj_resolution/2-1, obj_resolution/2-1,  sigma_obj, sigma_obj)(xx,yy) )
    return obj1[:-1,:-1]


def ObjectTilted(amp=1.0, x_0=0.0, y_0=0.0, s_x=1.0, s_y=1.0, ang=0.0):
    if s_x < 1e-3 or s_y < 1e-3: return None

    obj_resolution = 15
    lin_space = np.arange(-obj_resolution//2+1, obj_resolution//2+1)
    xx, yy = np.meshgrid(lin_space, lin_space)

    ang1 = ang*np.pi/180.0
    A =  np.cos(ang1)**2 / (2*s_x**2) + np.sin(ang1)**2 / (2*s_y**2)
    B = -np.sin(2*ang1)  / (4*s_x**2) + np.sin(2*ang1)  / (4*s_y**2)
    C =  np.sin(ang1)**2 / (2*s_x**2) + np.cos(ang1)**2 / (2*s_y**2)
    return amp * np.exp(-(A*(xx-x_0)**2 + 2*B*(xx-x_0)*(yy-y_0) + C*(yy-y_0)**2))


def NoisyPSF(tel, PSF, integrate=True):
    return tel.det.getFrame(PSF, noise=True, integrate=integrate)


# %%

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass
else:
    def symshow(x, lim=None, fixed=False, ax=None):
        x_ = np.nan_to_num(x)
        c_lim = np.maximum(np.abs(x_.min()), x_.max())
        
        if lim is not None and not fixed:
            c_lim = np.minimum(c_lim, lim)
            
        elif lim is not None and fixed:
            c_lim = lim
        
        if ax is None:
            return plt.imshow(x, vmin=-c_lim, vmax=c_lim, origin='lower')
        else:
            return ax.imshow(x, vmin=-c_lim, vmax=c_lim, origin='lower')


    def draw_PSF_difference(inp_0, inp_1, diff, is_log=False, diff_clims=None, crop=None, colormap='viridis'):
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        fig, axs = plt.subplots(1,3)
        fig.set_size_inches(15,15)

        if crop is None:
            data_0 = np.copy(inp_0)
            data_1 = np.copy(inp_1)
        else:
            data_0 = inp_0[crop]
            data_1 = inp_1[crop]

        if is_log:
            vmin = np.nanmin(np.log(data_0))
            vmax = np.nanmax(np.log(data_0))
            im0 = axs[0].imshow(np.log(data_0), vmin=vmin, vmax=vmax, cmap=colormap)
        else:
            vmin = np.nanmin(data_0)
            vmax = np.nanmax(data_0)
            im0 = axs[0].imshow(data_0, vmin=vmin, vmax=vmax, cmap=colormap)

        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes('right', size='10%', pad=0.05)
        axs[0].set_axis_off()
        fig.colorbar(im0, cax=cax, orientation='vertical')

        if is_log:
            im1 = axs[1].imshow(np.log(data_1), vmin=vmin, vmax=vmax, cmap=colormap)
        else:
            im1 = axs[1].imshow(data_1, vmin=vmin, vmax=vmax, cmap=colormap)
            
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes('right', size='10%', pad=0.05)
        axs[1].set_axis_off()
        fig.colorbar(im1, cax=cax, orientation='vertical')

        if diff_clims is None:
            diff_clims = np.abs(diff).max()

        im2 = axs[2].imshow(diff, cmap=plt.get_cmap("RdYlBu"), vmin=-diff_clims, vmax=diff_clims)
        divider = make_axes_locatable(axs[2])
        cax = divider.append_axes('right', size='10%', pad=0.05)
        axs[2].set_axis_off()
        fig.colorbar(im2, cax=cax, orientation='vertical')


try:
    from PIL import Image
    from matplotlib import cm
except ImportError:
    pass

else:
    def plt2PIL(fig=None):
        # Render the figure on a canvas
        if fig is None:
            # fig = plt.gcf()
            canvas = plt.get_current_fig_manager().canvas
        else:
            canvas = fig.canvas

        canvas.draw()
        rgba = canvas.buffer_rgba()

        # Create a numpy array from the bytes
        buffer = np.array(rgba).tobytes()
        # Create a PIL image from the bytes
        pil_image = Image.frombuffer('RGBA', (canvas.get_width_height()), buffer, 'raw', 'RGBA', 0, 1)

        return pil_image


    def save_GIF(array, duration=1e3, scale=1, path='test.gif', colormap=cm.viridis):
        from skimage.transform import rescale
        
        # If the input is an array or a tensor, we need to convert it to a list of PIL images first
        if type(array) == np.ndarray:
            gif_anim = []
            array_ = array.copy()

            if array.shape[0] != array.shape[1] and array.shape[1] == array.shape[2]:
                array_ = array_.transpose(1,2,0)

            for layer in np.rollaxis(array_, 2):
                buf = layer/layer.max()
                if scale != 1.0:
                    buf = rescale(buf, scale, order=0)
                gif_anim.append( Image.fromarray(np.uint8(colormap(buf)*255)) )
        else:
            # In this case, we can directly save the list of PIL images
            gif_anim = array

        # gif_anim[0].save(path, save_all=True, append_images=gif_anim[1:], optimize=False, quality=100, duration=duration, loop=0)
        gif_anim[0].save(path, save_all=True, append_images=gif_anim[1:], optimize=False, compress_level=0, duration=duration, loop=0)


    def save_GIF_RGB(images_stack, duration=1e3, downscale=4, path='test.gif'):
        from PIL import Image
        gif_anim = []
        
        def remove_transparency(img, bg_colour=(255, 255, 255)):
            alpha = img.convert('RGBA').split()[-1]
            bg = Image.new("RGBA", img.size, bg_colour + (255,))
            bg.paste(img, mask=alpha)
            return bg
        
        for layer in images_stack:
            im = Image.fromarray(np.uint8(layer*255))
            im.thumbnail((im.size[0]//downscale, im.size[1]//downscale), Image.ANTIALIAS)
            gif_anim.append( remove_transparency(im) )
            gif_anim[0].save(path, save_all=True, append_images=gif_anim[1:], optimize=True, duration=duration, loop=0)


try:
    from photutils.centroids import centroid_quadratic
    from photutils.profiles  import RadialProfile
    
except ImportError:
    pass

else:
    def calc_profile(data, xycen=None):
        if xycen is None:
            xycen = centroid_quadratic(np.abs(data))

        edge_radii = np.arange(data.shape[-1]//2)
        rp = RadialProfile(data, xycen, edge_radii)
        return rp.profile