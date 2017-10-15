'''
Created on 14. sep. 2017 v2

@author: LJB
'''
from __future__ import division
from numba import jit, float64, void
import numpy as np
import matplotlib.pyplot as plt
import timeit
import scipy.linalg as linalg


_DPI = 250
_SIZE = 0.7


def figure_set_default_size():
    F = plt.gcf()
    DefaultSize = [8 * _SIZE, 6 * _SIZE]
    print "Default size in Inches", DefaultSize
    print "Which should result in a %i x %i Image" % (_DPI * DefaultSize[0],
                                                      _DPI * DefaultSize[1])
    F.set_size_inches(DefaultSize)
    return F


def my_savefig(filename):
    F = figure_set_default_size()
    F.tight_layout()
    F.savefig(filename, dpi=_DPI)


def f(x):
    return 100 * np.exp(-10 * x)


def u(x):
    return 1 - (1 - np.exp(-10)) * x - np.exp(-10 * x)

@jit(void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:]))
def conda_tridiagonal_solve(u, temp, a, b, c, f):
    '''
    Solving a tridiagonal matrix equation [a, b, c]*u = f
    '''
    n = len(f)

    # decomposition and forward substitution
    btemp = b[0]
    u[0] = f[0] / btemp
    for i in xrange(1, n):
        temp[i] = c[i - 1] / btemp
        btemp = b[i] - a[i - 1] * temp[i]
        u[i] = (f[i] - a[i - 1] * u[i - 1]) / btemp

    # backwards substitution
    for i in xrange(n - 2, -1, -1):
        u[i] -= temp[i + 1] * u[i + 1]


def tridiagonal_solve(a, b, c, f):
    '''
    Solving a tridiagonal matrix equation [a, b, c]*u = f
    '''
    a, b, c, f = np.atleast_1d(a, b, c, f)
    n = len(f)
    u = np.zeros(n)
    temp = np.zeros(n)
    conda_tridiagonal_solve(u, temp, a, b, c, f)

    return u


def tridiagonal_solve_specific(f):
    '''
    Solving a tridiagonal matrix equation [-1, 2, -1]*u = f
    '''
    f = np.atleast_1d(f)
    n = len(f)
    u = np.zeros(n)
    temp = np.zeros(n)
    conda_tridiagonal_solve_spesific(u, temp, f)
    return u


@jit(void(float64[:], float64[:], float64[:]))
def conda_tridiagonal_solve_spesific(u, temp, f):
    '''
    Solving a tridiagonal matrix equation [a, b, c]*u = f
    '''

    n = len(f)
    # decomposition and forward substitution
    btemp = 2.0
    u[0] = f[0] / btemp
    for i in xrange(1, n):
        temp[i] = -1 / btemp
        btemp = 2 + temp[i]
        u[i] = (f[i] + u[i - 1]) / btemp

    # backwards substitution
    for i in xrange(n - 2, -1, -1):
        u[i] -= temp[i + 1] * u[i + 1]
    
# def tridiagonal_solve(a, b, c, f):
#     '''
#     Solving a tridiagonal matrix equation [a, b, c]*u = f
#     '''
#     a, b, c, f = np.atleast_1d(a, b, c, f)
#     n = len(f)
#     u = np.zeros(n)
#     temp = np.zeros(n)
# 
#     # decomposition and forward substitution
#     btemp = b[0]
#     u[0] = f[0] / btemp
#     for i in range(1, n):
#         temp[i] = c[i - 1] / btemp
#         btemp = b[i] - a[i - 1] * temp[i]
#         u[i] = (f[i] - a[i - 1] * u[i - 1]) / btemp
# 
#     # backwards substitution
#     for i in range(n - 2, -1, -1):
#         u[i] -= temp[i + 1] * u[i + 1]
#     return u
# 
# 
# def tridiagonal_solve_specific(f):
#     '''
#     Solving a tridiagonal matrix equation [a, b, c]*u = f
#     '''
#     f = np.atleast_1d(f)
#     n = len(f)
#     u = np.zeros(n)
#     temp = np.zeros(n)
# 
#     # decomposition and forward substitution
#     btemp = 2
#     u[0] = f[0] / btemp
#     for i in range(1, n):
#         temp[i] = -1 / btemp
#         btemp = 2 + temp[i]
#         u[i] = (f[i] + u[i - 1]) / btemp
# 
#     # backwards substitution
#     for i in range(n - 2, -1, -1):
#         u[i] -= temp[i + 1] * u[i + 1]
#     return u


def cpu_time(repetition=10, n=10**6):
    '''
    Grid n =10^6 and two repetitions gave an average of 7.62825565887 seconds.
    '''
    print timeit.timeit('solve_poisson({})'.format(n),setup='from __main__ import solve_poisson', number=repetition)/repetition


def cpu_time_specific(repetition=10, n=10**6):
    '''
    Grid n =10^6 and two repetitions gave an average of 7.512299363 seconds.
    '''
    print timeit.timeit('solve_poisson_specific({})'.format(n),setup='from __main__ import solve_poisson_specific', number=repetition)/repetition


def cpu_time_lu_solve(repetition=10, n=10):
    '''
    Grid n =10^6 and two repetitions gave an average of 7.512299363 seconds.
    '''
    print timeit.timeit('solve_poisson_with_lu({})'.format(n),setup='from __main__ import solve_poisson_with_lu', number=repetition)/repetition



def solve_poisson(n):
    h = 1.0 / (n + 1) # step_length
    x = np.arange(1, n + 1) * h
    fx = f(x)
    a = -np.ones(n)
    b = 2 * np.ones(n)
    c = -np.ones(n)
    v = tridiagonal_solve(a, b, c, fx * h ** 2)
    return x, v


def solve_poisson_specific(n):
    h = 1.0 / (n + 1) # step_length
    x = np.arange(1, n + 1) * h
    fx = f(x)
    v = tridiagonal_solve_specific(fx * h ** 2)
    return x, v

def solve_poisson_with_lu(n):
    h = 1.0 / (n + 1) # step_length
    x = np.arange(1, n + 1) * h
    fx = f(x)
    a = -np.ones(n-1)
    b = 2 * np.ones(n)
    c = -np.ones(n-1)
    A = np.diag(a, -1)+np.diag(b, 0)+np.diag(c,1)
    v = lu_solve(A, fx * h ** 2)
    return x, v

def test_compare():
    n = 10
    for i, n in enumerate([10, 100, 1000]):
        x, v = solve_poisson_specific(n)
        plt.figure(i)
        plt.plot(x, v, '.', label='numerical')
        plt.plot(x, u(x), label='exact')
        plt.title('n={}'.format(n))
        plt.legend()
        filename = "task1b_n{}.png".format(n)
        my_savefig(filename)


def error_test():
    max_error = []
    problem_sizes = np.array([10, 100, 1000, 10000, 100000, 1000000, 10000000])
    for i, n in enumerate(problem_sizes):
        x, v = solve_poisson_specific(n)
        ui = u(x)
        max_error.append(np.log10(max(np.abs((v-ui)/ui))))
    h = 1/(problem_sizes+1)
    plt.plot(np.log10(h), max_error)
    plt.title('Relative error')
    plt.xlabel('log10(h)')
    plt.ylabel('log10(relative error)')
    filename = 'error.png'
    my_savefig(filename)
    

def lu_solve(A, f):
    '''
    Solves A*u= f
    '''
    A, f = np.atleast_1d(A, f)
    lu_and_pivot = linalg.lu_factor(A)
    u = linalg.lu_solve(lu_and_pivot, f)
    return u
    
    
def lu_test_compare():
    n = 10
    for i, n in enumerate([10, 100, 1000]):
        x, v = solve_poisson_with_lu(n)
        plt.figure(i)
        plt.plot(x, v, '.', label='numerical')
        plt.plot(x, u(x), label='exact')
        plt.title('n={}'.format(n))
        plt.legend()

    
if __name__ == '__main__':
    #cpu_time(2)
    #cpu_time_specific(2)
    #test_compare()
    #solve_poisson_with_lu(10)
    #error_test()
    #lu_test_compare()
    for n in [10, 100, 1000, 2000]:
        cpu_time_specific(10, n)
        cpu_time(10,n)
        cpu_time_lu_solve(10, n)
        
        
    plt.show()