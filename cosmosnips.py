#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from scipy.special import hyp2f1
from scipy.optimize import fsolve
import warnings

def z2x(z, Om=0.31):
    """flat ΛCDM comoving distance χ in Mpc/h
    """
    z = np.asarray(z)
    OL = 1 - Om
    a = 1 / (1+z)
    return 2997.92458 * OL**-.5 * (hyp2f1(1/3, 1/2, 4/3, -Om*a**-3/OL) / a
                                - hyp2f1(1/3, 1/2, 4/3, -Om/OL))

def x2z(x, Om=0.31):
    """Inverse of z2x()
    """
    x = np.asarray(x)
    if any(xi >= z2x(1e3) for xi in x):
        warnings.warn("χ too large, touching the last scattering surface")
    z = fsolve(lambda y: z2x(y, Om=Om) - x, x / 2997.92458)
    return z

def z2t(z, Om=0.31):
    """flat ΛCDM age in yr/h
    """
    z = np.asarray(z)
    OL = 1 - Om
    a = 1 / (1+z)
    return 9.78e9 * 2/3 / np.sqrt(OL) * np.arcsinh(np.sqrt(OL/Om * a**3))

def D(z, Om=0.31, norm='MD'):
    """flat ΛCDM growth function

    Parameters
    ----------
    norm : str, optional
        - 'MD' (default): normalized to a at matter-dominated era
        - '0': normalized to 1 at redshift zero
    """
    z = np.asarray(z)
    OL = 1 - Om
    a = 1 / (1+z)
    if norm == 'MD':
        return a * hyp2f1(1, 1/3, 11/6, -OL*a**3/Om)
    elif norm == '0':
        return a * hyp2f1(1, 1/3, 11/6, -OL*a**3/Om) \
                / hyp2f1(1, 1/3, 11/6, -OL/Om)
    else:
        raise ValueError("unknown norm type")

def f(z, Om=0.31):
    """flat ΛCDM growth rate
    """
    z = np.asarray(z)
    OL = 1 - Om
    a = 1 / (1+z)
    aa3 = OL*a**3/Om
    return 1 - 6*aa3/11 * hyp2f1(2, 4/3, 17/6, -aa3) \
            / hyp2f1(1, 1/3, 11/6, -aa3)

def aH(z, Om=0.31):
    """flat ΛCDM comoving Hubble aH/c in h/Mpc
    """
    z = np.asarray(z)
    OL = 1 - Om
    a = 1 / (1+z)
    return np.sqrt(Om / a + OL * a*a) / 2997.92458

def rdz2x(r, d, z, axis=0):
    """RA, Dec, z to Cartesian coordinates, flat cosmology
    """
    r = np.deg2rad(r)
    d = np.deg2rad(d)
    xi = z2x(z)
    x = np.stack((xi * np.cos(d) * np.cos(r),
                  xi * np.cos(d) * np.sin(r),
                  xi * np.sin(d)), axis=axis)
    return x
