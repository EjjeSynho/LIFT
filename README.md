# LInearized Focal-plane Technique (LIFT)

The Python implementation of LIFT, a technique initially presented in [1]. The code is optimized for both CPU and GPU. For a detailed example of usage please refer to *'tests/example.ipynb'*.

## Required modules
```
numpy  
scipy
cupy (optional, to enable GPU acceleration)
pillow (optional)
photutils (optional)
matplotlib (optional)
skimage (optional)
```

## Acknowledgments
This code was developed during a master's thesis internship and a PhD program of Arseniy Kuznetsov, funded by ESO, LAM, and ONERA.
The original architecture of the code is inspired by the OOPAO (see https://github.com/cheritier/OOPAO).
Some parts of the Zernike modal basis implementation are copied from AOtools (see https://github.com/AOtools/aotools)

## References
[1]	S. Meimon, T. Fusco, and L. M. Mugnier, *LIFT: a focal-plane wavefront sensor for real-time low-order sensing on faint sources,* Opt. Lett., vol. 35, no. 18, p. 3036, Sep. 2010, doi: [10.1364/OL.35.003036.](https://doi.org/10.1364/OL.35.003036)

[2]	C. Plantet, S. Meimon, J.-M. Conan, and T. Fusco, *Experimental validation of LIFT for estimation of low-order modes in low-flux wavefront sensing,* Opt. Express, vol. 21, no. 14, p. 16337, Jul. 2013, doi: [10.1364/OE.21.016337.](https://doi.org/10.1364/OE.21.016337)

[3]	A. Kuznetsov, S. Oberti, C. Heritier, C. Plantet, B. Neichel, T. Fusco, S. Ströbele, C. Correia, *Study of the LIFT focal-plane wavefront sensor for GALACSI NFM,* Proceedings SPIE Adaptive Optics
Systems VIII, Jul. 2022, Montréal, Canada. pp.109, [hal-03796122.](https://hal.science/hal-03796122)

[4] G. Agapito, L. Busoni, G. Carlà, C. Plantet, S. Esposito, and P. Ciliegi, *MAORY/MORFEO and LIFT: can the low order wavefront sensors become phasing sensors?,* in Adaptive Optics Systems VIII, Aug. 2022, p. 195, doi: [10.1117/12.2629352.](https://doi.org/10.1117/12.2629352)

## License
This project is licensed under the terms of the MIT license.



