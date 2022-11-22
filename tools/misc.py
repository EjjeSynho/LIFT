import numpy as np
import matplotlib.pyplot as plt
try:
    import cupy as cp
except ImportError or ModuleNotFoundError:
    print('CuPy is not found, using NumPy backend...')
    cp = np


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


def mask_circle(N, r):
    factor = 0.5 * (1-N%2)
    coord_range = np.linspace(-N//2+N%2+factor, N//2-factor, N)
    xx, yy = np.meshgrid(coord_range, coord_range)
    pupil_round = np.zeros([N, N], dtype=np.int32)
    pupil_round[np.sqrt(yy**2 + xx**2) < r] = 1
    return pupil_round


# To make this function work, one must ensure that size of inp can be divided by N
def binning(inp, N):
    if N == 1: return inp
    xp = cp if hasattr(inp, 'device') else np
    out = xp.dstack(xp.split(xp.dstack(xp.split(inp, inp.shape[0]//N, axis=0)), inp.shape[1]//N, axis=1))
    return out.sum(axis=(0,1)).reshape([inp.shape[0]//N, inp.shape[1]//N]).T


def Gaussian2DTilted(amp=1.0, x_0=0.0, y_0=0.0, s_x=1.0, s_y=1.0, ang=0.0):
    if s_x < 1e-2 or s_y < 1e-2: return None

    obj_resolution = 15
    lin_space = np.arange(-obj_resolution//2+1, obj_resolution//2+1)
    xx, yy = np.meshgrid(lin_space, lin_space)

    ang1 = ang*np.pi/180.0
    A =  np.cos(ang1)**2 / (2*s_x**2) + np.sin(ang1)**2 / (2*s_y**2)
    B = -np.sin(2*ang1)  / (4*s_x**2) + np.sin(2*ang1)  / (4*s_y**2)
    C =  np.sin(ang1)**2 / (2*s_x**2) + np.cos(ang1)**2 / (2*s_y**2)
    return amp * np.exp(-(A*(xx-x_0)**2 + 2*B*(xx-x_0)*(yy-y_0) + C*(yy-y_0)**2))


def magnitudeFromPSF(tel, photons, band, sampling_time=None):
    if sampling_time is None:
        sampling_time = tel.det.sampling_time
    zero_point = tel.src.PhotometricParameters(band)[2]
    fluxMap = photons / tel.pupil.sum() * tel.pupil
    nPhoton = np.nansum(fluxMap / tel.pupilReflectivity) / (np.pi*(tel.D/2)**2) / sampling_time
    return -2.5 * np.log10(368 * nPhoton / zero_point )


def TruePhotonsFromMag(tel, mag, band, sampling_time): # [photons/aperture] !not per m2!
    c = tel.pupilReflectivity*np.pi*(tel.D/2)**2*sampling_time
    return tel.src.PhotometricParameters(band)[2]/368 * 10**(-mag/2.5) * c


def NoisyPSF(tel, PSF, integrate=True):
    return tel.det.getFrame(PSF, noise=True, integrate=integrate)


Nph_diff = lambda m_1, m_2: 10**(-(m_2-m_1)/2.5)
mag_diff = lambda ph_diff: -2.5*np.log10(ph_diff)