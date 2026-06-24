#%%
import numpy as np
try:
    import cupy as cp
    global_gpu_flag = True

except ImportError or ModuleNotFoundError:
    cp  = np

class Source:   #   v-- is a list of turples, each turple is: (band, magnitude)
    def __init__(self, input=None):
        if input is None:
            print('Error: empty input!')
            return
        
        self.__InitPhotometry()
        flux_from_mag = lambda mag, params: params[2]/368 * 10**(-0.4*mag)

        self.spectrum = []
        for wavelength, magnitude in input:
            photometric_params = self.PhotometricParameters(wavelength)
            num_photons = flux_from_mag(magnitude, photometric_params)
            self.spectrum.append({
                'wavelength': photometric_params[0],  # [m]
                'bandwidth' : photometric_params[1],  # [m]
                'flux'      : num_photons             # [photon/m2/s]
                })

        self.OPD = None
        self.tag = 'source'


    def __mul__(self, x):
        if hasattr(x, 'tag'):
            if x.tag == 'telescope': x.src = self
            return x
        elif isinstance(x, int) or isinstance(x, float) or isinstance(x, cp.ndarray) or isinstance(x, np.ndarray):
            for point in self.spectrum: point['flux'] *= x
            return self


    def __InitPhotometry(self):
        # photometry object [wavelength, bandwidth, zeroPoint]
        self.bands = {
            'U'   : [ 0.360e-6 , 0.070e-6 , 2.0e12 ],
            'B'   : [ 0.440e-6 , 0.100e-6 , 5.4e12 ],
            'V0'  : [ 0.500e-6 , 0.090e-6 , 3.3e12 ],
            'V'   : [ 0.550e-6 , 0.090e-6 , 3.3e12 ],
            'R'   : [ 0.640e-6 , 0.150e-6 , 4.0e12 ],
            'I'   : [ 0.790e-6 , 0.150e-6 , 2.7e12 ],
            'I1'  : [ 0.700e-6 , 0.033e-6 , 2.7e12 ],
            'I2'  : [ 0.750e-6 , 0.033e-6 , 2.7e12 ],
            'I3'  : [ 0.800e-6 , 0.033e-6 , 2.7e12 ],
            'I4'  : [ 0.700e-6 , 0.100e-6 , 2.7e12 ],
            'I5'  : [ 0.850e-6 , 0.100e-6 , 2.7e12 ],
            'I6'  : [ 1.000e-6 , 0.100e-6 , 2.7e12 ],
            'I7'  : [ 0.850e-6 , 0.300e-6 , 2.7e12 ],
            'R2'  : [ 0.650e-6 , 0.300e-6 , 7.92e12],
            'R3'  : [ 0.600e-6 , 0.300e-6 , 7.92e12],
            'R4'  : [ 0.670e-6 , 0.300e-6 , 7.92e12],
            'I8'  : [ 0.750e-6 , 0.100e-6 , 2.7e12 ],
            'I9'  : [ 0.850e-6 , 0.300e-6 , 7.36e12],
            'J'   : [ 1.215e-6 , 0.260e-6 , 1.9e12 ],
            'H'   : [ 1.654e-6 , 0.290e-6 , 1.1e12 ],
            'Kp'  : [ 2.1245e-6, 0.351e-6 , 6e11   ],
            'Ks'  : [ 2.157e-6 , 0.320e-6 , 5.5e11 ],
            'K'   : [ 2.179e-6 , 0.410e-6 , 7.0e11 ],
            'L'   : [ 3.547e-6 , 0.570e-6 , 2.5e11 ],
            'M'   : [ 4.769e-6 , 0.450e-6 , 8.4e10 ],
            'Na'  : [ 0.589e-6 , 0        , 3.3e12 ],
            'EOS' : [ 1.064e-6 , 0        , 3.3e12 ]
        }
        # Build interpolation grid from unique, non-zero-bandwidth entries only.
        # Duplicate centre wavelengths (e.g. I1/I4, I2/I8, I5/I7/I9) and zero-
        # bandwidth entries (Na, EOS) would cause division-by-zero or nonsensical
        # interpolated bandwidths when PhotometricParameters is called with a float.
        seen = set()
        interp_entries = []
        for v in self.bands.values():
            wl = v[0]
            if v[1] > 0 and wl not in seen:
                seen.add(wl)
                interp_entries.append(v)
        interp_entries.sort(key=lambda v: v[0])
        self.__interp_bands = interp_entries          # list of [wl, bw, zp], unique & sorted
        self.__wavelengths  = np.array([v[0] for v in interp_entries])


    def PhotometricParameters(self, inp):
        if isinstance(inp, str):
            if inp not in self.bands.keys():
                print('Error: there is no band with the name "'+inp+'"')
                return None
            else:
                return self.bands[inp]

        elif isinstance(inp, float):    # perform interpolation of parameters for a current wavelength
            if inp < self.__wavelengths.min() or inp > self.__wavelengths.max():
                print('Error: specified value is outside the defined wavelength range!')
                return None

            # Find the two bracketing entries in the deduplicated, sorted grid.
            idx = np.searchsorted(self.__wavelengths, inp)
            idx = np.clip(idx, 1, len(self.__wavelengths) - 1)
            p_1 = np.array(self.__interp_bands[idx - 1])
            p_2 = np.array(self.__interp_bands[idx])
            l_1, l_2 = p_1[0], p_2[0]
            weight = (inp - l_1) / (l_2 - l_1)

            return weight * (p_2 - p_1) + p_1

        else:
            print('Incorrect input: "'+inp+'"')
            return None             


    def GetSpectrum(self):
        data = []
        for point in self.spectrum:
            data_point = [v for _,v in point.items()]
            data.append( [data_point[0], data_point[2]] )
        return np.array(data)
