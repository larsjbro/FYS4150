'''
Created on 8. des. 2017

@author: ljb
'''
from __future__ import division, absolute_import
from numba import jit, float64, void
import numpy as np
import matplotlib.pyplot as plt
import timeit
import scipy.linalg as linalg
from fys4150.project5.figure_saver import my_savefig


def fun(x):
    return 100 * np.exp(-10 * x)


def u(x):
    return 1 - (1 - np.exp(-10)) * x - np.exp(-10 * x)


@jit(void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:]))
def conda_tridiagonal_solve(x, temp, a, b, c, f):
    '''
    Solving a tridiagonal matrix equation [a, b, c]*x = f
    '''
    n = len(f)

    # decomposition and forward substitution
    btemp = b[0]
    x[0] = f[0] / btemp
    for i in xrange(1, n):
        temp[i] = c[i - 1] / btemp
        btemp = b[i] - a[i - 1] * temp[i]
        x[i] = (f[i] - a[i - 1] * x[i - 1]) / btemp

    # backwards substitution
    for i in xrange(n - 2, -1, -1):
        x[i] -= temp[i + 1] * x[i + 1]


def tridiagonal_solve(a, b, c, f):
    '''
    Solving a tridiagonal matrix equation [a, b, c]*x = f
    '''
    a, b, c, f = np.atleast_1d(a, b, c, f)
    n = len(f)
    x = np.zeros(n)
    temp = np.zeros(n)
    conda_tridiagonal_solve(x, temp, a, b, c, f)
    return x


def tridiagonal_solve_periodic(a, b, c, f):
    '''
    Solving a periodic tridiagonal matrix equation

         A * x = f
     where
         diag(A) = [b0, b1, b2, ...., bn-1]
         diag(A, -1) = [a0, a1, a2, ...., an-2]
         diag(A, 1) = [c0, c1, c2, ...., cn-2]

         A[0, n-1] = an-1   (periodic boundary conditions)
         A[n-1, 0] = cn-1   (periodic boundary conditions)

    Reference
    ---------
    https://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    '''
    a, b, c, f = np.atleast_1d(a, b, c, f)
    n = len(f)
    y = np.zeros(n)
    q = np.zeros(n)
    temp = np.zeros(n)
    b1 = np.copy(b)
    b0 = b1[0]
    cn1 = c[n - 1]
    an1 = a[n - 1]
    b1[0] = b0 * 2
    b1[n - 1] += cn1 * an1 / b0
    u = np.zeros(n)
    u[0] = -b0
    u[-1] = cn1
    v = np.zeros(n)
    v[0] = 1
    v[-1] = - an1 / b0
    conda_tridiagonal_solve(y, temp, a, b1, c, f)
    conda_tridiagonal_solve(q, temp, a, b1, c, u)
    numerator = np.dot(v, y)
    denominator = (1 + np.dot(v, q))
    if denominator==0:
        scale=1.49
    else:
        scale = numerator / denominator
    return y - q * scale


def tridiagonal_solve_specific_periodic(f):
    '''
    Solving a periodic tridiagonal matrix equation
         A * x = f
     where
         diag(A) = [2,2,2, ...., 2]
         diag(A, -1) = diag(A, 1) =[-1, -1, -1, ...., -1]
         A[0, n-1] = A[n-1, 0] = -1  (periodic boundary conditions)

    Reference
    ---------
    https://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)

    '''
    f = np.atleast_1d(f)
    n = len(f)
    a = -np.ones(n)
    b = 2 * np.ones(n)
    c = -np.ones(n)
    return tridiagonal_solve_periodic(a, b, c, f)


def tridiagonal_solve_specific(b):
    '''
    Solving a tridiagonal matrix equation [-1, 2, -1]*x = b
    '''
    b = np.atleast_1d(b)
    n = len(b)
    x = np.zeros(n)
    temp = np.zeros(n)
    conda_tridiagonal_solve_spesific(x, temp, b)
    return x


@jit(void(float64[:], float64[:], float64[:]))
def conda_tridiagonal_solve_spesific(x, temp, b):
    '''
    Solving a tridiagonal matrix equation [-1, 2, -1]*x = b
    '''

    n = len(b)
    # decomposition and forward substitution
    btemp = 2.0
    x[0] = b[0] / btemp
    for i in xrange(1, n):
        temp[i] = -1.0 / btemp
        btemp = 2.0 + temp[i]
        x[i] = (b[i] + x[i - 1]) / btemp

    # backwards substitution
    for i in xrange(n - 2, -1, -1):
        x[i] -= temp[i + 1] * x[i + 1]


def cpu_time(repetition=10, n=10**6):
    '''
    Grid n =10^6 and two repetitions gave an average of 7.62825565887 seconds.
    '''
    t = timeit.timeit('solve_poisson_general({})'.format(n),
                      setup='from __main__ import solve_poisson_general',
                      number=repetition) / repetition
    print(t)
    return t


def cpu_time_specific(repetition=10, n=10**6):
    '''
    Grid n =10^6 and two repetitions gave an average of 7.512299363 seconds.
    '''
    t = timeit.timeit('solve_poisson_specific({})'.format(n),
                      setup='from __main__ import solve_poisson_specific',
                      number=repetition) / repetition
    print(t)
    return t


def cpu_time_lu_solve(repetition=10, n=10):
    '''
    Grid n =10^6 and two repetitions gave an average of 7.512299363 seconds.
    '''
    t = timeit.timeit('solve_poisson_with_lu({})'.format(n),
                      setup='from __main__ import solve_poisson_with_lu',
                      number=repetition) / repetition

    print(t)
    return t


def solve_poisson_general(n):
    h = 1.0 / (n + 1)  # step_length
    x = np.arange(1, n + 1) * h
    fx = fun(x)
    a = -np.ones(n)
    b = 2 * np.ones(n)
    c = -np.ones(n)
    v = tridiagonal_solve(a, b, c, fx * h ** 2)
    return x, v


def solve_poisson_specific(n):
    h = 1.0 / (n + 1)  # step_length
    x = np.arange(1, n + 1) * h
    fx = fun(x)

    v = tridiagonal_solve_specific(fx * h ** 2)
    return x, v

def solve_specific_periodic_with_lu(f):
    f = np.atleast_1d(f)
    n = len(f)
    a = -np.ones(n - 1)
    b = 2 * np.ones(n)
    c = -np.ones(n - 1)
    A = np.diag(a, -1) + np.diag(b, 0) + np.diag(c, 1)
    A[0, -1] -= 1
    A[-1, 0] -= 1
    v = lu_solve(A, f)
    return v

def solve_poisson_with_lu(n):
    h = 1.0 / (n + 1)  # step_length
    x = np.arange(1, n + 1) * h
    fx = fun(x)
    a = -np.ones(n - 1)
    b = 2 * np.ones(n)
    c = -np.ones(n - 1)
    A = np.diag(a, -1) + np.diag(b, 0) + np.diag(c, 1)
    v = lu_solve(A, fx * h ** 2)
    return x, v


def tri_solve_test_compare():
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
    for n in problem_sizes:
        x, v = solve_poisson_specific(n)
        ui = u(x)
        max_error.append(np.log10(max(np.abs((v - ui) / ui))))
    h = 1 / (problem_sizes + 1)
    plt.plot(np.log10(h), max_error)
    plt.title('Relative error')
    plt.xlabel('log10(h)')
    plt.ylabel('log10(relative error)')
    filename = 'error.png'
    my_savefig(filename)


def lu_solve(A, b):
    '''
    Solves A*x= b
    '''
    A, b = np.atleast_1d(A, b)
    lu_and_pivot = linalg.lu_factor(A)
    x = linalg.lu_solve(lu_and_pivot, b)
    return x


def lu_test_compare():
    n = 10
    for i, n in enumerate([10, 100, 1000]):
        x, v = solve_poisson_with_lu(n)
        plt.figure(i)
        plt.plot(x, v, '.', label='numerical')
        plt.plot(x, u(x), label='exact')
        plt.title('n={}'.format(n))
        plt.legend()


def plot_run_times(sizes=(10, 100, 1000, 2000)):
    times = np.zeros((len(sizes), 3))
    for i, n in enumerate(sizes):
        times[i, 0] = cpu_time_specific(10, n)
        times[i, 1] = cpu_time(10, n)
        times[i, 2] = cpu_time_lu_solve(10, n)

    for i, name in enumerate(['tri_spec', 'tri', 'lu']):
        plt.loglog(sizes, times[:, i], label=name)
    plt.legend()


if __name__ == '__main__':
    # cpu_time(2)
    # cpu_time_specific(2)
    #     tri_solve_test_compare()
    # solve_poisson_with_lu(10)
    # error_test()
    # lu_test_compare()
    # plot_run_times(sizes)
    plt.show()
