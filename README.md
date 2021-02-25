# SFA OAM Circular
Using the SFA to compute the orbital angular momentum of a photoelectron ionized via a strong circular sin^2 pulse laser field.

To build on linux use the build.sh script, run:
`./build.sh` in the directory

If not on linux run the python commands inside the build script.

You must have python and cython installed for this to work.


The data processing and plotting can be found in the python notebook. It is recommend to use jupyter lab and anaconda to view this.

## SFACircPulse.pyx information
A '.pyx' file is a cython file that uses the flexibility of python but with parts that have the speed of c++.

The SFACircPulse.pyx file codes a python class called SFACircularPulse. As the name suggests it implements the SFA for a circular polarized laser field with a sin^2 pulse.

### Class member functions
The only accessible member functions from outside the class are those with cpdef  or def in front of them. Those with only cdef are internal and can not be called outside the class. The most important functions are:

- Afx(t) vector potential in x direction
- Afy(t) vector potential in y direction
- Efx(t) electric field in x direction
- Efy(t) electric field in y direciton
- S(p, theta, phi, t) semi-classical action
- TimesGen(p, theta, phi) find all times of ionization via the saddle point approximation for a specific momentum coordinate
- M(p, theta, phi, tf) Transition amplitude for a final momentum point computed using the saddle point approximaton
- Ml(p, theta, Nphi) output vector of size Nphi of OAM dependent transition amplitude using FFT of M(p, theta, phi, tf)

**To be implemented:**

- d0(p, theta, phi, t) function for the matrix element incorporating the effect of the bound state

### Class member variables
There are the following member variables relevant to the input parameters of the problem and as ways to control the class.

The following are class member that are set when you make an instance and remain fixed after that point (readonly)

- Ip: the ionization potential
- Up: Poneromotive energy
- omega: carrier frequency
- N: number of lasser cycles
- phi: carrier envelope phase (currently obsolete as it is not implemented)


Other member varibles include
constant complex iminary unit I, constant Pi and the square root of Pi
