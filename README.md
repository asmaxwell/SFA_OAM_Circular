# SFA OAM Circular
Using the SFA to compute the orbital angular momentum of a photoelectron ionized via a strong circular sin^2 pulse laser field.

To build on linux use the build.sh script, run:
`./build.sh` in the directory

If not on linux run the python commands inside the build script.

You must have python and cython installed for this to work.


The data processing and plotting can be found in the python notebook. It is recommend to use jupyter lap and anaconda to view this.

## SFACircPulse.pyx information
A '.pyx' file is a cython file that uses the flexibility of python but with parts that have the speed of c++.

The SFACircPulse.pyx file code a python class called SFACircularPulse. As the name suggests it implements the SFA for a circular polarized laser field with a sin^2 pulse.

### Class member functions
The only accessible member funciton from outside the class are those with cpdef in front of them. Those with only cdef are internal and can not be called outside the class.

### Class member variables
There are the following member varibles relvant to the input parameters of the problem and as ways to control the class:
Ip: the ionization potential
Up: Poneromotive energy
omega: carrier frequency
N: number of lasser cycles
phi: carrier envelope phase (currently obsolete as it is not implemented)


Other member varibles include
constant complex iminary unit I, constant Pi and the square root of Pi
