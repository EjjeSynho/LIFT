import numpy as np

try:
    import cupy as cp
    global_gpu_flag = True
except (ImportError, ModuleNotFoundError):
    cp = np
    global_gpu_flag = False


class LWE_modes:
    """
    Low-Wind-Effect / VLT petal modal basis.

    This class does NOT try to recover petals by connected-component labelling.
    Instead, it expects the petal-mode cube produced by e.g.

        PupilVLT(samples, petal_modes=True)

    whose convention is assumed to be

        [petal pistons..., petal tips..., petal tilts...]

    along the last axis, as in VLT_pupil.py.

    The stored basis follows the same convention as Zernike.modesFullRes:

        [height, width, nModes]

    Each mode can optionally be renormalized to unit RMS over the union pupil.
    """

    def __init__(self, modes_num=None, normalize=True, input_order="vlt"):
        """
        Parameters
        ----------
        modes_num : int or None
            Number of LWE modes to keep. If None, all supplied modes are used.
        normalize : bool
            If True, normalize every mode to unit RMS over the union pupil.
            This is usually convenient when mixing with Zernikes.
        input_order : {'vlt', 'old'}
            Interpretation of the 3*N input modes.

            'vlt': [pistons, tips, tilts], matching PupilVLT(..., petal_modes=True).
            'old': [pistons, tilts, tips], matching the previous BuildPetalBasis order.
        """
        global global_gpu_flag

        if input_order not in ("vlt", "old"):
            raise ValueError("input_order must be either 'vlt' or 'old'.")

        self.nModes = modes_num
        self.nPetals = None
        self.modesFullRes = None
        self.pupil = None
        self.coefs = None
        self.modes_names = []
        self.normalize = normalize
        self.input_order = input_order
        self.gpu = global_gpu_flag

    @property
    def gpu(self):
        return self.__gpu

    @gpu.setter
    def gpu(self, var):
        if var:
            self.__gpu = True
            if hasattr(self, "modesFullRes") and self.modesFullRes is not None:
                if not hasattr(self.modesFullRes, "device"):
                    self.modesFullRes = cp.array(self.modesFullRes, dtype=cp.float32)
            if hasattr(self, "coefs") and self.coefs is not None:
                if not hasattr(self.coefs, "device"):
                    self.coefs = cp.array(self.coefs, dtype=cp.float32)
        else:
            self.__gpu = False
            if hasattr(self, "modesFullRes") and self.modesFullRes is not None:
                if hasattr(self.modesFullRes, "device"):
                    self.modesFullRes = self.modesFullRes.get()
            if hasattr(self, "coefs") and self.coefs is not None:
                if hasattr(self.coefs, "device"):
                    self.coefs = self.coefs.get()

    @staticmethod
    def _as_numpy(x):
        if hasattr(x, "get"):
            return x.get()
        return np.asarray(x)

    @staticmethod
    def _to_hwn_modes(modes):
        """
        Return modes as [H, W, M]. Accept [H, W, M] or [M, H, W].
        """
        modes = LWE_modes._as_numpy(modes).astype(np.float64, copy=False)

        if modes.ndim != 3:
            raise ValueError(
                "petal_modes must be a 3D array with shape [H, W, M] or [M, H, W]."
            )

        # PupilVLT returns [H, W, M]. Keep it.
        if modes.shape[0] == modes.shape[1]:
            return modes

        # Common alternative: [M, H, W].
        if modes.shape[1] == modes.shape[2]:
            return np.moveaxis(modes, 0, -1)

        raise ValueError(
            f"Could not infer petal_modes layout from shape {modes.shape}; "
            "expected [H, W, M] or [M, H, W]."
        )

    @staticmethod
    def _infer_petal_count(n_modes):
        if n_modes % 3 != 0:
            raise ValueError(
                f"The petal-mode cube has {n_modes} modes, but LWE expects 3*N modes: "
                "pistons, tips, tilts."
            )
        return n_modes // 3

    @staticmethod
    def _normalize_modes_to_unit_rms(modes, pupil):
        """
        Normalize each [H, W] mode to unit RMS over `pupil > 0`.
        """
        modes = modes.copy()
        mask = pupil > 0

        if mask.sum() == 0:
            raise ValueError("The inferred/explicit LWE pupil is empty.")

        flat = modes[mask, :]
        rms = np.sqrt(np.mean(flat**2, axis=0))

        bad = np.where(~np.isfinite(rms) | (rms == 0))[0]
        if bad.size > 0:
            raise ValueError(f"Some LWE modes have zero/non-finite RMS: {bad.tolist()}")

        modes /= rms[None, None, :]
        return modes, rms

    def _make_mode_names(self):
        if self.input_order == "vlt":
            second_block = "tip"
            third_block = "tilt"
        else:
            second_block = "tilt"
            third_block = "tip"

        names = []
        for i in range(self.nPetals):
            names.append(f"Petal {i + 1} piston")
        for i in range(self.nPetals):
            names.append(f"Petal {i + 1} {second_block}")
        for i in range(self.nPetals):
            names.append(f"Petal {i + 1} {third_block}")
        self.modes_names = names

    def computeLWE(self, tel=None, petal_modes=None, pupil=None, normalize=None):
        """
        Compute/store the LWE modal basis from externally supplied petal modes.

        Parameters
        ----------
        tel : Telescope-like object, optional
            Used only for GPU-state compatibility and, if `pupil` is not given,
            as a fallback source of the pupil mask.
        petal_modes : ndarray, required unless `tel.petal_modes` exists
            Petal-mode cube, preferably returned by
            `PupilVLT(samples, petal_modes=True)`.
            Shape can be [H, W, 3*N] or [3*N, H, W].
        pupil : ndarray, optional
            Explicit pupil mask. If omitted, the union of the piston-petal block
            is used. If `tel` is given and has `.pupil`, that can also be used,
            but the union of supplied petals is usually safer for LWE.
        normalize : bool or None
            Override `self.normalize` for this computation.

        Notes
        -----
        Unlike the old implementation, this method does not call scipy.label and
        does not split a binary pupil into connected islands. The petal geometry
        and local tip/tilt modes must be supplied explicitly.
        """
        if tel is not None:
            self.gpu = self.gpu and getattr(tel, "gpu", False)

        if petal_modes is None:
            if tel is not None and hasattr(tel, "petal_modes"):
                petal_modes = tel.petal_modes
            else:
                raise ValueError(
                    "`petal_modes` must be provided. For VLT use "
                    "PupilVLT(samples, petal_modes=True)."
                )

        modes = self._to_hwn_modes(petal_modes)
        total_modes_available = modes.shape[-1]
        self.nPetals = self._infer_petal_count(total_modes_available)

        if pupil is None:
            # Prefer the union of the explicitly provided piston petals.
            pistons = modes[:, :, :self.nPetals]
            self.pupil = (np.sum(np.abs(pistons), axis=-1) > 0).astype(np.float64)
        else:
            self.pupil = self._as_numpy(pupil).astype(np.float64, copy=False)

        if self.pupil.shape != modes.shape[:2]:
            raise ValueError(
                f"pupil shape {self.pupil.shape} does not match petal_modes spatial "
                f"shape {modes.shape[:2]}."
            )

        do_normalize = self.normalize if normalize is None else normalize
        if do_normalize:
            modes, _ = self._normalize_modes_to_unit_rms(modes, self.pupil)

        if self.nModes is None:
            self.nModes = total_modes_available

        if self.nModes > total_modes_available:
            raise ValueError(
                f"Requested {self.nModes} LWE modes, but only {total_modes_available} "
                f"are available for {self.nPetals} petals."
            )

        modes = modes[:, :, :self.nModes]
        self._make_mode_names()

        coefs = np.zeros(total_modes_available, dtype=np.float64)
        coefs[:self.nPetals] = 1.0
        self.coefs = coefs[:self.nModes]

        self.modesFullRes = modes

        if self.gpu:
            self.modesFullRes = cp.array(self.modesFullRes, dtype=cp.float32)
            self.coefs = cp.array(self.coefs, dtype=cp.float32)
            self.pupil = cp.array(self.pupil, dtype=cp.float32)

    def modeName(self, index):
        if index < 0:
            return "Incorrect index!"
        elif index >= len(self.modes_names):
            return f"LWE {index}"
        else:
            return self.modes_names[index]

    def wavefrontFromModes(self, tel, coefs_inp):
        """
        Generate wavefront shape corresponding to given LWE coefficients.
        """
        xp = cp if self.gpu else np

        coefs = xp.array(coefs_inp).flatten()
        coefs[xp.where(xp.abs(coefs) < 1e-13)] = xp.nan
        valid_ids = xp.where(xp.isfinite(coefs))[0]

        if self.modesFullRes is None:
            raise RuntimeError(
                "LWE modes were not computed. Call computeLWE(..., petal_modes=...) first."
            )

        if self.nModes < coefs.shape[0]:
            raise ValueError(
                f"Coefficient vector has length {coefs.shape[0]}, "
                f"but only {self.nModes} LWE modes were computed."
            )

        return self.modesFullRes[:, :, valid_ids] @ coefs[valid_ids]

    def Mode(self, coef):
        if self.modesFullRes is None:
            raise RuntimeError("LWE modes were not computed yet.")
        return self.modesFullRes[:, :, coef]
