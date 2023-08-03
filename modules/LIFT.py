import numpy as np
from scipy import linalg as lg
from scipy import signal as sg

try:
    import cupy as cp
    from cupyx.scipy import signal as csg
    global_gpu_flag = True

except ImportError or ModuleNotFoundError:
    print('CuPy is not found, using NumPy backend...')
    cp  = np
    csg = sg
    global_gpu_flag = False

from tools.misc import binning


class LIFT:
    def __init__(self, tel, modeBasis, diversity_OPD, iterations):
        global global_gpu_flag
        self.tel             = tel
        self.modeBasis       = modeBasis
        self.diversity_OPD   = diversity_OPD
        self.iterations      = iterations
        self.gpu             = self.tel.gpu and global_gpu_flag

        if self.gpu:
            self.diversity_OPD = cp.array(self.diversity_OPD, dtype=cp.float32)


    def print_modes(self, A_vec):
        xp = cp if self.gpu else np
        for i in range(A_vec.shape[0]):
            val = A_vec[i]
            if val != None and not xp.isnan(val): val = xp.round(val, 4)
            print('Mode #', i, val)


    def obj_convolve(self, mat):
        xg = csg if self.gpu else sg
        if self.tel.object is not None:
            return xg.convolve2d(mat, self.tel.object, boundary='symm', mode='same') / self.tel.object.sum()
        else: return mat


    def generateLIFTinteractionMatrices(self, coefs, modes_ids, flux_norm=1.0, numerical=False):
        xp = cp  if self.gpu else np

        if isinstance(coefs, list):
            coefs = xp.array(coefs)

        initial_OPD = self.modeBasis.wavefrontFromModes(self.tel, coefs) + self.diversity_OPD
        
        H = []
        if not numerical:
            for point in self.tel.src.spectrum:
                wavelength = point['wavelength']

                initial_amplitude = xp.sqrt(self.tel.flux(point['flux']*flux_norm, self.tel.det.sampling_time)) * self.tel.pupil
                k = 2*xp.pi/wavelength

                initial_phase = k * initial_OPD
                Pd = xp.conj( self.tel.PropagateField(initial_amplitude, initial_phase, wavelength, return_intensity=False) )
                
                H_spectral = []
                for i in modes_ids:
                    buf = self.tel.PropagateField(self.modeBasis.modesFullRes[:,:,i]*initial_amplitude, initial_phase, wavelength, return_intensity=False)
                    derivative = 2*binning((xp.real(1j*buf*Pd)), self.tel.oversampling) * k
                    derivative = self.obj_convolve(derivative)

                    H_spectral.append(derivative.flatten())   
                H.append(xp.vstack(H_spectral).T)
        else:
            delta = 1e-9 # [nm]
            for point in self.tel.src.spectrum:
                H_spectral = []
                for i in modes_ids:
                    self.tel.src.OPD = (self.modeBasis.modesFullRes[:,:,i] * delta) + initial_OPD
                    tmp1 = self.tel.ComputePSF() * flux_norm           

                    self.tel.src.OPD = -(self.modeBasis.modesFullRes[:,:,i] * delta) + initial_OPD
                    tmp2 = self.tel.ComputePSF() * flux_norm

                    derivative = self.obj_convolve((tmp1-tmp2)/2/delta)

                    H_spectral.append( derivative.flatten() )
                H.append(np.vstack(H_spectral).T)
        
        return xp.dstack(H).sum(axis=2) # sum all spectral interaction matricies


    def Reconstruct(self, PSF_inp, R_n, mode_ids, A_0=None, verbous=False, optimize_norm='sum'):
        """
        Function to reconstruct modal coefficients from the input PSF image using LIFT

        Parameters:
            PSF (ndarray):                   2-d array of the input PSF image to reconstruct.

            R_n (ndarray or string or None): The pixel weighting matrix for LIFT. It can be passed to the function.
                                             from outside, modeled ('model'), updated dynamically ('iterative'), or
                                             assumed to be just detector's readout noise ('None').
            mode_ids (ndarray or list):      IDs of the modal coefficients to be reconstructed
            A_0 (ndarray):                   initial assumtion for the coefficient values. In some sense, it acts as an additional 
                                             phase diversity on top of the main phase diversity which is passed when class is initialized.

            A_ref (ndarray):                 Reference coefficients to compare the reconstruction with. Useful only when ground-truth (A_ref) is known.

            verbous (bool):                  Set 'True' to print the intermediate reconstruction results.

            optimize_norm (string or None):  Recomputes the flux of the recontructed PSF iteratively. If 'None', the flux is not recomputed, 
                                             this is recommended only if the target brightness is precisely known. In mosyt of the case it is 
                                             recommended to switch it on. When 'sum', the reconstructed PSF is normalized to the sum of the pixels 
                                             of the input PSF. If 'max', then the reconstructed PSF is normalized to the maximal value of the input PSF.
        """
        if self.gpu:
            xp = cp
            convert = lambda x: cp.asnumpy(x)
        else:
            xp = np
            convert = lambda x: x

        def PSF_from_coefs(coefs):
            OPD = self.modeBasis.wavefrontFromModes(self.tel, coefs)
            self.tel.src.OPD = self.diversity_OPD + OPD
            return self.obj_convolve( self.tel.ComputePSF() )
                   
        C      = []  # optimization criterion
        Hs     = []  # interaction matrices for every iteration
        P_MLs  = []  # estimators for every iteration
        A_ests = []  # history of estimated coefficients
        
        PSF = xp.array(PSF_inp, dtype=xp.float32)
        modes = xp.sort( xp.array(mode_ids, dtype=xp.int32) )

        # Account for the intial assumtion for the coefficients values
        if A_0 is None:
            A_est = xp.zeros(modes.max().item()+1, dtype=xp.float32)
        else:
            A_est = xp.array(A_0, dtype=xp.float32)
        A_ests.append(xp.copy(A_est))

        def normalize_PSF(PSF_in):
            if optimize_norm is not None and optimize_norm is not False:
                if optimize_norm == 'max': return (PSF_in/PSF_in.max(), PSF_in.max())
                if optimize_norm == 'sum': return (PSF_in/PSF_in.sum(), PSF_in.sum())
            else: return (PSF_in, 1.0)
        
        PSF_0, flux_cap = normalize_PSF(PSF_from_coefs(A_est)) # initial PSF assumtion, normalized to 1.0

        flux_scale = 1.0
        if optimize_norm is not None and optimize_norm is not False:
            if   optimize_norm == 'max': flux_scale = PSF.max()
            elif optimize_norm == 'sum': flux_scale = PSF.sum()
        
        PSF_cap = xp.copy(PSF_0) * flux_scale

        criterion  = lambda i: xp.abs(C[i]-C[i-1]) / C[i]
        coefs_norm = lambda v: xp.linalg.norm(v[modes], ord=2)

        def inverse_Rn(Rn):
            return 1./xp.clip(Rn.flatten(), a_min=1e-6, a_max=Rn.max())  

        if R_n is not None:
            if isinstance(R_n, str): #basically if it's 'model' or 'iterative':
                inv_R_n = inverse_Rn(PSF_0*flux_scale + self.tel.det.readoutNoise**2)
            else:
                inv_R_n = inverse_Rn(xp.array(R_n))
        else:
            inv_R_n = inverse_Rn(xp.ones_like(PSF)*self.tel.det.readoutNoise**2)

        for i in range(self.iterations):    
            dI = (PSF-PSF_cap).flatten()
    
            C.append( xp.dot(dI*inv_R_n, dI) )  # check the convergence
            if i > 0 and (criterion(i)<1e-6 or coefs_norm(A_ests[i]-A_ests[i-1])<1e-12):
                if verbous:
                    print('Criterion', criterion(i), 'is reached at iter.', i)
                break
            
            # Generate interaction matricies
            H = self.generateLIFTinteractionMatrices(A_est, modes, flux_scale/flux_cap)                               

            # Maximum likelyhood estimation
            P_ML = xp.linalg.pinv(H.T * inv_R_n @ H) @ H.T * inv_R_n
            d_A = P_ML @ dI
            A_est[modes] += d_A
            
            # Save the intermediate results for history
            Hs.append(H)
            P_MLs.append(P_ML)
            A_ests.append(xp.copy(A_est))

            if verbous:
                print('Criterion:', criterion(i))
                self.print_modes(d_A)
                print()
            
            # Update the PSF image with the estimated coefficients
            PSF_cap, flux_cap = normalize_PSF(PSF_from_coefs(A_est))
            PSF_cap *= flux_scale
            
            if isinstance(R_n, str):
                if R_n == 'iterative':
                    inv_R_n = inverse_Rn(PSF_cap + self.tel.det.readoutNoise**2)

        history = { # contains intermediate data saved at every iteration
            'P_ML' : convert( xp.dstack(P_MLs) ),
            'H'    : convert( xp.dstack(Hs) ),
            'A_est': convert( xp.squeeze(xp.dstack(A_ests), axis=0) ),
            'C'    : convert( xp.array(C) )
        }
        return convert(A_est), convert(PSF_cap), history
