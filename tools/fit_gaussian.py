# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp( -(((center_x-x)/width_x)**2 + ((center_y-y)/width_y)**2)/2 )


def gaussian_fourier(height, center_u, center_v, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda u,v: height*np.exp( -(((center_u-u)*width_x)**2 + ((center_v-v)*width_y)**2) * 2 * np.pi**2 )


def moments(data):
    """
        Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution by calculating its
        moments
    """
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-y)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y


def fitgaussian(data):
    """
        Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution found by a fit
    """
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape))-data)
    p, success = optimize.leastsq(errorfunction, params)
    return p


def plot_gaussian(data, params, extents=None):
    fit = gaussian(*params)

    plt.contour(fit(*np.indices(data.shape)), cmap=plt.cm.copper, extent=extents)
    ax = plt.gca()
    (height, x, y, width_x, width_y) = params

    FWHM_x = 2*np.sqrt(2*np.log(2)) * width_x
    FWHM_y = 2*np.sqrt(2*np.log(2)) * width_y

    plt.text(0.95, 0.05, """
    x : %.1f
    y : %.1f
    FWHM$_x$ : %.1f
    FWHM$_y$ : %.1f""" %(x, y, FWHM_x, FWHM_y), fontsize=16, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)


def FitAndPlotGauss1D(x, y, label=None):
    n = len(x)                          #the number of data
    A = y.max()
    w = y / y.max()

    mean  = np.sum(x*w) / np.sum(w)
    sigma = np.sqrt( sum(w*(x-mean)**2) / ( (n-1)/n * np.sum(w) ) )
        
    def gaus(x, a, x0, sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))

    popt, pcov = optimize.curve_fit(gaus, x, y, p0=[A,mean,sigma])
    #plt.plot(x, gaus(x, *popt), 'r:', label=label)    
    #print("Sigma:", popt[2])
    return popt

