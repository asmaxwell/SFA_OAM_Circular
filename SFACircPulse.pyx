# distutils: language = c++
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Pulse version of the SFA for circular polarization
"""
Created on Tuesday Nov 24 10:29:00 2020

@author: asmaxwell

SFA pulse class
"""
#import scipy.optimize as op
import scipy.integrate as it
import scipy.special as sp
cimport scipy.special.cython_special as csp
import functools
import multiprocessing

import numpy as np
cimport numpy as np
from numpy cimport ndarray

cimport cython

from libcpp cimport bool

from libc.math cimport sin as sin_re
#from libc.math cimport sinc as sinc_re
from libc.math cimport cos as cos_re
from libc.math cimport acos as acos_re
from libc.math cimport exp as exp_re
from libc.math cimport sqrt as sqrt_re
from libc.math cimport abs as abs_re
from libc.math cimport pow as pow_re
from libc.math cimport atan2 as atan2_re
#from libc.math cimport  cyl_bessel_j as bessel_j_re


#fused types
ctypedef fused dbl_or_cmplx:
    double
    double complex
    
ctypedef fused c_dbl_int:
    int
    double
    double complex
    
cdef extern from "<complex.h>" namespace "std" nogil:
    double complex exp(double complex z)
    double complex sin(double complex z)
    double complex cos(double complex z)
    double complex sqrt(double complex z)
    double complex acos(double complex z)
    double complex log(double complex z)
    #double complex pow(double complex z, c_dbl_int z)
    double real(double complex z)
    double imag(double complex z)


cdef double complex I = 1j
cdef double Pi = np.pi
cdef double rtPi = np.sqrt(Pi)
cdef int cacheSize = 2**20

### shorcut functions to efficntly switch trig between real and complex varients    
cdef sin_c(dbl_or_cmplx t):
        if(dbl_or_cmplx is double):
            return sin_re(t)
        else:
            return sin(t)
cdef cos_c(dbl_or_cmplx t):
        if(dbl_or_cmplx is double):
            return cos_re(t)
        else:
            return cos(t)
        
cdef cot_c(dbl_or_cmplx t):
        return cos_c(t)/sin_c(t)




cdef class SFACircularPulse:
    '''
        Class to compute the transition amplitude M(p) and its dervative M_g(p) using the SFA and saddle point approximation
    '''
    #memeber variables like in C++!
    cdef readonly double Ip, Up, rt2Up, omega, T, sgn
    cdef readonly int N, Target
    cdef readonly bool anticlockwise
    #cdef object __weakref__ # enable weak referencing support
    
    def __init__(self, Ip_ = 0.5, Up_ = 0.44, omega_ = 0.057, N_ = 6
                 , phi_ = 0., anticlockwise_=True, TCycles_ = 0, Target_ = 0):
        '''
            Initialise field and target parameters defaults correspond to 800nm wl and 2 10^14 W/cm^2 intensity
        '''
        #Target parameters
        self.Ip = Ip_
        self.Target = Target_
        
        #Pulse parameters
        self.Up = Up_
        self.rt2Up = np.sqrt(2*Up_) #must change this if Up is changed! Fixed by making Up readonly
        self.omega = omega_
        self.N = N_
        self.anticlockwise = anticlockwise_ #direction of field rotation
        cdef double Cycles = 2*Pi /self.omega
        self.T = Cycles * TCycles_ #delay parameter, input (TCycles) in number of cycle of delay
        
        #Pulse parameters for speed
        #Fill in if they crop up
        self.sgn = 1.0 if self.anticlockwise else -1.0
        #print("anticlockwise = ",self.anticlockwise," sgn = ", self.sgn)

    #@functools.lru_cache(maxsize=cacheSize)
    cdef dbl_or_cmplx F(s, dbl_or_cmplx t):
        '''
        envelope for Sin^2 laser pulse
        '''
        #need to be fast can get evalutated millions of times
        if(real(t)<0 or real(t)>2*s.N*Pi/s.omega):
            return 0
        Fval = -s.rt2Up * sin_c(s.omega * t / (2*s.N))**2
        return Fval
    
    #@functools.lru_cache(maxsize=cacheSize)
    cpdef dbl_or_cmplx Afx(s, dbl_or_cmplx t):
        '''
        VVector potential for Sin^2 circular laser pulse in x component
        '''
        if(real(t)<0 or real(t)>2*s.N*Pi/s.omega):
            return 0
        cdef dbl_or_cmplx envelope, carrier
        envelope = s.F(t)
        if(dbl_or_cmplx is double):
            carrier = cos_re(s.omega*t)
        else:
            carrier = cos(s.omega*t)
        return  envelope * carrier
    
    #@functools.lru_cache(maxsize=cacheSize)
    cpdef dbl_or_cmplx Afy(s, dbl_or_cmplx t):
        '''
        Vector potential for Sin^2 circular laser pulse in y component
        '''
        if(real(t)<0 or real(t)>2*s.N*Pi/s.omega):
            return 0
        cdef dbl_or_cmplx envelope, carrier
        envelope = s.F(t)
        if(dbl_or_cmplx is double):
            carrier = s.sgn * sin_re(s.omega*t)
        else:
            carrier = s.sgn * sin(s.omega*t)
        return  envelope * carrier
    
    #@functools.lru_cache(maxsize=cacheSize)
    cpdef dbl_or_cmplx Efx(s, dbl_or_cmplx t):
        '''
        Electric Field for Sin^2 laser pulse
        '''
        if(real(t)<0 or real(t)>2*s.N*Pi/s.omega):
            return 0
        if(dbl_or_cmplx is double):
            fac = s.rt2Up*s.omega*sin_re(s.omega*t/(2*s.N))/s.N
            a1 = cos_re(s.omega*t)*cos_re(s.omega*t/(2*s.N))
            a2 = sin_re(s.omega*t)*sin_re(s.omega*t/(2*s.N))
            
        else:
            fac = s.rt2Up*s.omega*sin(s.omega*t/(2*s.N))/s.N
            a1 = cos(s.omega*t)*cos(s.omega*t/(2*s.N))
            a2 = sin(s.omega*t)*sin(s.omega*t/(2*s.N))
        
        return fac*(a1-s.N*a2)
    #@functools.lru_cache(maxsize=cacheSize)
    cpdef dbl_or_cmplx Efy(s, dbl_or_cmplx t):
        '''
        Electric Field for Sin^2 laser pulse
        '''
        if(real(t)<0 or real(t)>2*s.N*Pi/s.omega):
            return 0
        if(dbl_or_cmplx is double):
            fac = s.rt2Up*s.omega*sin_re(s.omega*t/(2*s.N))/s.N
            a1 = sin_re(s.omega*t)*cos_re(s.omega*t/(2*s.N))
            a2 = cos_re(s.omega*t)*sin_re(s.omega*t/(2*s.N))
            
        else:
            fac = s.rt2Up*s.omega*sin(s.omega*t/(2*s.N))/s.N
            a1 = sin(s.omega*t)*cos(s.omega*t/(2*s.N))
            a2 = cos(s.omega*t)*sin(s.omega*t/(2*s.N))
        
        return s.sgn * fac*(a1+s.N*a2)

    
    
    #@functools.lru_cache(maxsize=cacheSize)
    cpdef dbl_or_cmplx AfxI(s, dbl_or_cmplx t):
        '''
            Integral of vector potential in x component
        '''
        cdef double factor
        if(s.N==1):
            return -(sqrt(s.Up)*(2*t*s.omega - 4*sin_c(t*s.omega) + sin_c(2*t*s.omega)))/(4.*sqrt(2)*s.omega)
        else:
            factor = -s.rt2Up/(2*s.omega*(1-s.N*s.N))
            if(dbl_or_cmplx is double):
                simpleTrig = s.N*cos_re(s.omega*t)*sin_re(s.omega*t/s.N)
                longTrig = (-1 + s.N*s.N - s.N*s.N*cos_re(s.omega*t/s.N))*sin_re(s.omega*t)
            else:
                simpleTrig = s.N*cos(s.omega*t)*sin(s.omega*t/s.N)
                longTrig = (-1 + s.N*s.N - s.N*s.N*cos(s.omega*t/s.N))*sin(s.omega*t)

            return factor * (longTrig + simpleTrig)
    
    #@functools.lru_cache(maxsize=cacheSize)
    cpdef dbl_or_cmplx AfyI(s, dbl_or_cmplx t):
        '''
            Integral of vector potential y component
        '''
        cdef double factor
        if(s.N==1):
            return  s.sgn *(sqrt(2)*sqrt(s.Up)*sin_c((t*s.omega)/2.)**4)/s.omega
        else:
            factor = -s.rt2Up/(2*s.omega*(1-s.N*s.N))
            if(dbl_or_cmplx is double):
                simpleTrig = s.N*sin_re(s.omega*t)*sin_re(s.omega*t/s.N)
                longTrig = (1 - s.N*s.N + s.N*s.N*cos_re(s.omega*t/s.N))*cos_re(s.omega*t)
            else:
                simpleTrig =s.N*sin(s.omega*t)*sin(s.omega*t/s.N)
                longTrig = (1 - s.N*s.N + s.N*s.N*cos(s.omega*t/s.N))*cos(s.omega*t)

            return  s.sgn * factor * (longTrig + simpleTrig - 1)

    

    #@functools.lru_cache(maxsize=cacheSize)
    cpdef dbl_or_cmplx Af2I(s, dbl_or_cmplx t):
        '''
            Integral of vector potential squared
        '''
        cdef double c0 = -12*s.N*Pi
        if(dbl_or_cmplx is double):
            out = c0 - 8*s.N*sin_re(s.omega*t/s.N) + s.N*sin_re(2*s.omega*t/s.N) + 6*t*s.omega
        else:
            out = c0 - 8*s.N*sin(s.omega*t/s.N) + s.N*sin(2*s.omega*t/s.N) + 6*t*s.omega
            
        return -(s.Up/(8*s.omega))*out
        
        
    #@functools.lru_cache(maxsize=cacheSize)    
    cpdef dbl_or_cmplx S(s, double p, double theta, double phi, dbl_or_cmplx t):
        '''
            Action as given by the SFA for a Pulse
        '''
        cdef dbl_or_cmplx tTerms = (s.Ip + 0.5 * p*p )*t #- Pi*s.N*p*p/s.omega
        cdef dbl_or_cmplx linAI = -p*sin_re(theta)*(cos_re(phi) * s.AfxI(t) + sin_re(phi) * s.AfyI(t))
        cdef dbl_or_cmplx quadAI = -0.5*s.Af2I(t)
        cdef double delayTerm = (s.Ip + 0.5 * p*p )*s.T 
        return tTerms + linAI + quadAI + delayTerm
    
    
    cpdef dbl_or_cmplx DS(s, double p, double theta, double phi, dbl_or_cmplx t):
            cdef px = p*sin_re(theta)*cos_re(phi), py = p*sin_re(theta)*sin_re(phi)
            return s.Ip + 0.5*p**2 + 0.5*(s.Afx(t)*(2*px+s.Afx(t)) + s.Afy(t)*(2*py+s.Afy(t)))
    #@functools.lru_cache(maxsize=cacheSize)    
    cdef DSZ(s, double p, double theta, double phi):
        '''
            Derivative of the action tranformed by t->i N/omega Log[z] for esay solving
            This creates an 2(N+1) polynomial which can be efficeintly solved and the solutions easily 
            transformed back. This function passes the roots of the polynomial as numpy array so they can be solved using np.roots.          
            It is clear there will be N+1 solutions and their complex 
            conjugates.
        '''
        cdef double complex exp_phi = exp(-I*phi*s.sgn)
        cdef double CZ0 = p*s.rt2Up*sin_re(theta)/8
        cdef double CZNm1 = s.Up/16, CZN1 = s.Ip + 0.5*p*p + (3./8.)*s.Up
        
        #costruct polynomial in z of order 2*(N+1)
        poly_coeffs = np.zeros(2*s.N+3) + 0.*I
        #const terms
        poly_coeffs[0:3] = [CZ0/exp_phi , -2*CZ0/exp_phi, CZ0/exp_phi]
        #N order terms (+= accounts for cases where coefficients combine)
        poly_coeffs[(s.N-1):(s.N+4)] += [CZNm1, -4*CZNm1, CZN1, -4*CZNm1, CZNm1 ]
        #2N order term
        poly_coeffs[(2*s.N):] += [CZ0*exp_phi, -2*CZ0*exp_phi, CZ0*exp_phi]
        
        
        return poly_coeffs
    
    cpdef double complex DSZ_val(s, double p, double theta, double phi, dbl_or_cmplx z):
        poly_coeffs = s.DSZ(p, theta, phi)
        cdef double complex sum_val = 0
        for n in range(0, len(poly_coeffs)):
            sum_val += poly_coeffs[n] * z**n
        return sum_val
    
    cdef double complex addIfRealNeg(s, double complex ts):
        if(real(ts)<0):
            return ts + 2*Pi*s.N/s.omega
        else:
            return ts
        
    #@functools.lru_cache(maxsize=cacheSize)
    cpdef TimesGen(s, double p, double theta, double phi):
        '''
            Solution for times found by transforming the derivative of the action into
            a 2(N+1) polynomial and solving using np.roots. This should be a very effiecint way to get 
            all roots
        '''
        poly_coeffs = s.DSZ(p, theta, phi)
        z_roots = np.roots(poly_coeffs)
        #now we must transform back using t=I N Log[z]/omega
        ts_roots = I*s.N*np.log(z_roots)/s.omega
        ts_roots = [ts for ts in ts_roots if imag(ts)>0 ] #remove (divergent) solutions with negative imag
        #make sure all t values are in the domain [0, 2*pi*N/omega]
        ts_roots = [s.addIfRealNeg(ts) for ts in ts_roots]
        #sort real parts to easily select specific solutions        
        return sorted(ts_roots,  key=np.real)
        
    #1 varible determinant for saddle point approximation
    cpdef dbl_or_cmplx DDS(s, double p, double theta, double phi, dbl_or_cmplx t):
        cdef double px = p*sin_re(theta)*cos_re(phi), py = p*sin_re(theta)*sin_re(phi)
        return -s.Efx(t)*(px+s.Afx(t)) - s.Efy(t)*(py+s.Afy(t))
    
#prefactor
    #@functools.lru_cache(maxsize=cacheSize)
    cdef double complex d0(s, double p, double theta, double phi, double complex ts):
        '''
            Bound state prefactor <p+A(t)|HI(t)|0>, ground state defined by s.Target
        '''
        cdef double complex Efxt = s.Efx(ts)
        cdef double complex Efyt = s.Efy(ts)
        cdef double complex Afxt = s.Afx(ts)
        cdef double complex Afyt = s.Afy(ts)
        cdef double complex px = p*cos_re(theta) + Afxt
        cdef double complex py = p*sin_re(theta) + Afyt
        cdef double complex out = 1.
        if(s.Target==1):
            out = s.GaussianPot(px, py, Efxt, Efyt)
        else:
            out = 3.0*Pi*sqrt(I/s.DDS(p, theta, phi, ts))/pow_re(2.0, 3.5) 
                        
        return out
    
    cdef double complex GaussianPot(s, double complex px, double complex py, double complex Efxt, double complex Efyt):
        '''
            Prefactor for attoscience simulator project, using a Gaussian potential
            V0 and alpha are defined so that the ground state has and energy of -Ip
            The potential is given by V0*exp(-px*px/(alpha*alpha)) +V0*exp(-py*py/(alpha*alpha))
        '''       
        
        cdef double rtTwo = sqrt_re(2.0)
        cdef double V0 = s.Ip + 2./rtTwo
        cdef double fac = -sqrt_re(Pi)/sqrt_re(rtTwo)
        #here we use the fact that px*px + py*py = -2*s.Ip
        cdef double complex exps=exp(s.Ip/rtTwo)
        return Efxt * Efyt * fac * px*py * exps
    

    
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    #@functools.lru_cache(maxsize=cacheSize)
    cpdef double complex M(s, double p, double theta, double phi, double tf = np.inf):#double pz, double px, double t, int N, int eLim):
        '''
            Final transition amplitude
            Constructed as sum 
        '''
        #eLim is 1 or 2, the number of orbits accounted for
        cdef double complex MSum = 0.
        times = s.TimesGen(p, theta, phi)
        
        for ts in times:
            if(real(ts)<tf):
                det = sqrt(2*Pi*I/s.DDS(p, theta, phi, ts))
                expS = exp(I*s.S(p, theta, phi, ts))
                d0 = s.d0(p, theta, phi, ts)
                MSum += d0*det*expS
        return MSum
    #transition amplitude in cartesian co-ordinates
    cpdef double complex Mxy(s, px, py, pz, tf = np.inf):
        cdef double p = sqrt_re(px*px + py*py +pz*pz)
        cdef double theta = acos_re(pz/p)
        cdef double phi = atan2_re(py, px)
        return s.M(p, theta, phi, tf)
    #list comprehension over cartesian transition amplitude
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    def M_List(s, pxList, pyList, double pz, tf = np.inf):
        return np.array([s.Mxy(px, py, pz, tf) for px, py in zip(pxList, pyList)])
    
    cpdef Ml(s, double p, double theta, int Nphi = 250):
        '''
            This is the fourier series coeiffint of M to get the OAM distribusion.
            It is computed taking advantage of the FFT
        '''
        phiList = np.linspace(-Pi, Pi, Nphi)
        MphiList = [s.M(p, theta, phi) for phi in phiList]
        return np.fft.fft(MphiList)/Nphi
    
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    def Ml_List(s, pList, theta, Nphi = 250):
        return np.array([[abs(M)**2 for M in s.Ml(p, theta, Nphi)] for p in pList]).T

######   ---   Funcitons for the analytic monochromatic approximation to pulse   ---   #####
    

    cpdef Ml_Mono(s, double p, double theta, l):
        '''
                function to return the fully analytic monchromatic approximation to 
        '''
        cdef double E1 = (2*s.Ip+2*s.Up+p*p)
        #compute prefactor
        cdef double part1 = E1*E1/(4*p*p*sin_re(theta))
        cdef double complex DDS = p*s.omega*sqrt(2*s.Up-part1)*sin_re(theta)
        cdef double complex pref = sqrt(2*Pi*I/DDS)

        #compute action
        cdef double complex S0_1 = E1*acos(E1/(2*s.rt2Up*p*sin_re(theta))-1e-16 * I) #selecting +ive imaginary part branch of acos
        cdef double S0_2 = sqrt_re(E1*E1+4*p*p*s.Up*(cos_re(2*theta)-1))
        cdef double complex S0 = (S0_1 - I*S0_2)/(2*s.omega)

        #compute sinc
        sinc_val = np.sinc((E1-2*s.omega*l)/(2*s.omega))

        #compute ATI rings
        cdef double complex OM = (exp(I*s.N*Pi*E1/s.omega)-1)/(exp(I*Pi*E1/s.omega)-1)
        #print('OM = ',OM,', pref = ',pref,', S0 = ',S0_1/(2*s.omega),', sinc = ',sinc_val)
        return OM*pref*exp(I*S0)*sinc_val
        
    
######   ---   Here functions are defined for the analytical fourier series computation   ---   #####
    cdef double OAM_S(s, double p, double theta, int l, double t):
        '''
            Action for the analytical OAM action
        '''
        cdef double constTerms = (s.Ip + 0.5*p*p)*t
        cdef double quadTerms = 0 if (t<0 or t>2*s.N*Pi/s.omega) else -0.5*s.Af2I(t)
        cdef double trigTerm = 0 if (t<0 or t>2*s.N*Pi/s.omega) else atan2_re(s.AfxI(t), s.AfyI(t))*l
        
        return constTerms + quadTerms + trigTerm 
    
    cpdef double complex OAM_Integrand(s, double t, double p, double theta, int l):
        cdef double S1 = s.OAM_S(p, theta, l, t)
        cdef double complex exp_S = exp(I*S1)
        cdef AI_abs = 0 if (t<0 or t>2*s.N*Pi/s.omega) else sqrt_re(s.AfxI(t)**2 + s.AfyI(t)**2)
        cdef double BesJ = sp.jv(l, p*sin_re(theta)*AI_abs) 
        
        return BesJ*exp_S
    
    cpdef double OAM_IntegrandRe(s, double t, double p, double theta, int l):
        cdef double S1 = s.OAM_S(p, theta, l, t)
        cdef double cos_S = cos_re(S1)
        cdef AI_abs = 0 if (t<0 or t>2*s.N*Pi/s.omega) else sqrt_re(s.AfxI(t)**2 + s.AfyI(t)**2)
        cdef double BesJ = sp.jv(l, p*sin_re(theta)*AI_abs) 
        
        return BesJ*cos_S
    
    cpdef double OAM_IntegrandIm(s, double t, double p, double theta, int l):
        cdef double S1 = s.OAM_S(p, theta, l, t)
        cdef double sin_S = sin_re(S1)
        cdef AI_abs = 0 if (t<0 or t>2*s.N*Pi/s.omega) else sqrt_re(s.AfxI(t)**2 + s.AfyI(t)**2)
        cdef double BesJ = sp.jv(l, p*sin_re(theta)*AI_abs) 
        
        return BesJ*sin_S
    
    cpdef double complex OAM_Ml(s, double p, double theta, int l, double err = 1.0e-4, int limit = 2000):
        cdef double valRe, valIm, errorRe, errorIm
        valRe, errorRe = it.quad(s.OAM_IntegrandRe, 0, 2*s.N*Pi/s.omega, args = (p, theta, l), epsabs=err, epsrel=err, limit=limit  )
        valIm, errorIm = it.quad(s.OAM_IntegrandIm, 0, 2*s.N*Pi/s.omega, args = (p, theta, l), epsabs=err, epsrel=err, limit=limit  )
        #tList = np.linspace(0, 2*s.N*Pi/s.omega, Nt)
        #MtList = [s.OAM_Integrand(t, p, theta, l) for t in tList]
        
        return valRe + I*valIm#[valRe + I*valIm, errorRe + I*errorIm]#np.fft.fft(MphiList)/Nphi
    
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    def OAM_M_List(s, pList, theta, l, err = 1.0e-4, limit = 2000):
        return np.array([np.abs(s.OAM_Ml(p, theta, l, err, limit))**2 for p in pList])
    