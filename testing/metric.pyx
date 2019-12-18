from libc cimport abs as absc
cimport cython
cimport numpy as np

ctypedef double (*metric_ptr)(double[:], double[:])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.float64_t _calcQuatMetric(double[:] a, double[:] b):
    
#    return 1 - absc(a[0]*b[0] + a[1]*b[1]+ a[2]*b[2]+ a[3]*b[3])
    return 
    
@cython.boundscheck(False)
@cython.wraparound(False)
def calcQuatMetric(a, b):
    return _calcQuatMetric(a,b)
