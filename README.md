# simtod
TOD simulation with beam mismatch.

## Usage
1. Setup
```sh
python setup.py build_ext -if
```

```libhealpix_cxx``` and ```libopenblas``` are required during the setup.

2. Simple example
```python3
import _simtod
import numpy as np
import healpy as hp
#set mismatch parameter (dg, dx, dy, fwhm, dfwhm, dp, dc) dx, dy, fwhm, dfwhm are in arcmin unit
#the mismatch parameter are consistent with BICEP beam paper. arXiv:1904.01640
beam_para = (.05, 0.76, -.32, 19, .5, .002, 0)

# For now only TP Leakage is considered
Tmap = np.load('./tmap.npy')
nside = hp.get_nside(Tmap)

# Arbitrary scan strategy, 
theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
psi = np.zeros_like(theta)

# The _simtod.simtod function will return an numpy array with shape (2, nsample).
tod = _simtod.simtod(beam_para, Tmap, theta, phi, psi)
```
