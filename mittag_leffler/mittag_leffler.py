# -*- coding: utf-8 -*-
"""
Created on Tue Dec 7 2022
@author: davidsantiagoquevedo
"""
import numpy as np
from scipy.special import gamma as eb_gamma
from scipy.integrate import quad

def mittag_leffler_point(z, alpha, beta, inf = 100):
    """Mittag-Leffler function (MFF) for a real value z
   
    Parameters
    ----------
    z : float64
        Evaluation point
    alpha : float64
        alpha coefficient of the MFF function. Now limited to real values but should admit complex values
    beta : float64
        beta coefficient of the MFF function. Now limited to real values but should admit complex values
    inf : int64
        Sufficiently large value for the  summation on the definition (in general 100 is enough to vanis 1/gamma dependency)
    """
    k = np.arange(0, inf, 1, dtype=int)
    return (z**k/eb_gamma(alpha*k + beta)).sum()

def mittag_leffler_vector(z_vect, alpha, beta, inf = 100):
    """Mittag-Leffler function (MFF) for a real value z 1xn numpy array
    This is the most computational efficient way of evaluation 
    (See: https://stackoverflow.com/questions/35215161/most-efficient-way-to-map-function-over-numpy-array)
    
    Parameters
    ----------
    vect : 1xn numpy array 
        Evaluation vector
    alpha : float64
        alpha coefficient of the MFF function. Now limited to real values but should admit complex values
    beta : float64
        beta coefficient of the MFF function. Now limited to real values but should admit complex values
    inf : int64
        Sufficiently large value for the  summation on the definition (in general 100 is enough to vanis 1/gamma dependency)
    """
    mtlf = lambda x: mittag_leffler_point(x, alpha, beta, inf)
    v_mtlf = np.vectorize(mtlf)
    return v_mtlf(z_vect)

def mittag_leffler(z, alpha, beta, inf = 100):
    assert isinstance(z, (float, np.ndarray))
    if isinstance(z, float):
        return(mittag_leffler_point(z, alpha, beta, inf))
    elif isinstance(z, np.ndarray):
        return(mittag_leffler_vector(z, alpha, beta, inf))
    

def mittag_leffler_stable(z, alpha):
    """Stable Piecewise implementation for the Mittag-Leffler function
    (See: https://stackoverflow.com/questions/48645381/instability-in-mittag-leffler-function-using-numpy)
    
    Parameters
    ----------
    z : 1xn numpy array 
        Evaluation vector
    alpha : float64
        alpha coefficient of the MFF function. Now limited to real values but should admit complex values
    """
    z = np.atleast_1d(z)
    if alpha == 0:
        return 1/(1 - z)
    elif alpha == 1:
        return np.exp(z)
    elif alpha > 1 or all(z > 0):
        k = np.arange(100)
        return np.polynomial.polynomial.polyval(z, 1/eb_gamma(alpha*k + 1))

    # a helper for tricky case, from Gorenflo, Loutchko & Luchko
    def _MLf(z, alpha):
        if z < 0:
            f = lambda x: (np.exp(-x*(-z)**(1/alpha)) * x**(alpha-1)*np.sin(np.pi*alpha)
                          / (x**(2*alpha) + 2*x**alpha*np.cos(np.pi*alpha) + 1))
            return 1/np.pi * quad(f, 0, np.inf)[0]
        elif z == 0:
            return 1
        else:
            return mittag_leffler_stable(z, alpha)
    return np.vectorize(_MLf)(z, alpha)

def prabhakar_mittag_leffler_point(z, alpha, beta, gamma, inf = 100):
    """This function evaluates the Prabhakar Mittag-Leffler function (MFF) of at a real value z
   
    Parameters
    ----------
    z : float64
        Evaluation point
    alpha : float64
        a coefficient of the MFF function. Now limited to real values but should admit complex values
    beta : float64
        b coefficient of the MFF function. Now limited to real values but should admit complex values
    gamma : float64
        b coefficient of the MFF function. Now limited to real values but should admit complex values
    inf : int64
        Sufficiently large value for the  summation on the definition (in general 100 is enough to vanis 1/gamma dependency)
    """
    k = np.arange(0, inf, 1, dtype=int)
    return (z**k/eb_gamma(alpha*k + beta)*(eb_gamma(k+gamma)/eb_gamma(k+1))).sum()*(1/eb_gamma(gamma))

def prabhakar_mittag_leffler_vector(z_vect, alpha, beta, gamma, inf = 100):
    """This function evaluates the Mittag-Leffler function (MFF) in a 1xn numpy array
    
    Parameters
    ----------
    z_vect : 1xn numpy array 
        Evaluation vector
    a : float64
        a coefficient of the MFF function. Now limited to real values but should admit complex values
    b : float64
        b coefficient of the MFF function. Now limited to real values but should admit complex values
    r : float64
        b coefficient of the MFF function. Now limited to real values but should admit complex values
    inf : int64
        Sufficiently large value for the  summation on the definition (in general 100 is enough to vanis 1/gamma dependency)
    """
    pmtlf = lambda x: prabhakar_mittag_leffler_point(x, alpha, beta, gamma, inf)
    v_pmtlf = np.vectorize(pmtlf)
    return v_pmtlf(z_vect)

def prabhakar_mittag_leffler(z, alpha, beta, gamma, inf = 100):
    assert isinstance(z, (float, np.ndarray))
    if isinstance(z, float):
        return(prabhakar_mittag_leffler_point(z, alpha, beta, gamma, inf))
    elif isinstance(z, np.ndarray):
        return(prabhakar_mittag_leffler_vector(z, alpha, beta, gamma, inf))
    
if __name__ == "__main__":
  mittag_leffler()
  prabhakar_mittag_leffler()   
