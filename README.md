# simtod
TOD simulation with beam mismatch.

## Usage
1. Setup
```sh
python setup.py build_ext -if
```

```libhealpix_cxx``` and ```libopenblas``` are required during the setup.

2. Example
```python3
import _simtod
import numpy as np
import healpy as hp
#set mismatch parameter (dg, dx, dy, fwhm, dfwhm, dp, dc) 
beam_para = (.05, 0.76, -.32, 19, .5, .002, 0)

# Only TP Leakage is considered
Tmap = np.load('./tmap.npy')
nside = hp.get_nside(Tmap)

# Arbitrary scan strategy, 
theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
psi = np.zeros_like(theta)
_simtod.simtod(beam_para, Tmap, theta, phi, psi)
```
