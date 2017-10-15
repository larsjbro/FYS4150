'''
Created on 14. sep. 2017 v2

@author: LJB
'''
from __future__ import division, absolute_import
from numba import jit, float64, int64, void, int32
import numpy as np
import matplotlib.pyplot as plt
import timeit
import scipy.linalg as linalg
from matplotlib.pyplot import xlabel


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


def cpu_time(repetition=10, n=100):
    '''
    Grid n =10^6 and two repetitions gave an average of 7.62825565887 seconds.
    '''
    time_per_call = timeit.timeit('solve_schroedinger({},5)'.format(n),
                                  setup='from __main__ import solve_schroedinger',
                                  number=repetition) / repetition
    return time_per_call


def cpu_time_jacobi(repetition=10, n=100):
    '''
    Grid n =10^6 and two repetitions gave an average of 7.512299363 seconds.
    '''
    time_per_call = timeit.timeit('solve_schroedinger_jacobi({},5)'.format(n),
                                  setup='from __main__ import solve_schroedinger_jacobi',
                                  number=repetition) / repetition
    return time_per_call


def jacobi_method(A, epsilon=1.0e-8):  #2b,2d
    '''
    Jacobi's method for finding eigen_values
    eigenvectors of the symetric matrix A.
    The eigen_values of A will be on the diagonal
    of A, with eigenvalue i being A[i][i].
    The jth component of the ith eigenvector
    is stored in R[i][j].
    A: input matrix (n x n)
    R: empty matrix for eigenvectors (n x n)
    n: dimension of matrices
    7.4 Jacobi's method 219
    '''

    # Setting up the eigenvector matrix
    A = np.array(A)  # or A=np.atleast_2d(A)
    n = len(A)

    eigen_vectors = np.eye(n)

#     for i in range(n):
#         for j in range(n):
#             if i == j:
#                 eigen_vectors[i][j] = 1.0
#             else:
#                 eigen_vectors[i][j] = 0.0

    max_number_iterations = n**3
    iterations = 0

    max_value, k, l = max_off_diag(A)
    while max_value > epsilon and iterations < max_number_iterations:
        max_value, k, l = max_off_diag(A)
        rotate(A, eigen_vectors, k, l, n)
        iterations += 1

    # print "Number of iterations: {}".format(iterations)
    eigen_values = np.diag(A)  # eigen_values are the diagonal elements of A
    # return eigenvectors and eigen_values
    return eigen_vectors, eigen_values, iterations

# @jit(float64(float64[:, :], int32[1], int32[1]), nopython=True)


@jit(nopython=True)
def max_off_diag(A):
    ''' Function to find the maximum matrix element. Can you figure out a more
     elegant algorithm?'''
    n = len(A)
    max_val_out = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            absA = abs(A[i][j])
            if absA > max_val_out:
                max_val_out = absA
                l = i
                k = j
    return max_val_out, k, l


@jit(void(float64[:, :], float64[:, :], int64, int64, int64), nopython=True)
def rotate(A, R, k, l, n):
    '''Function to find the values of cos and sin'''

    if A[k][l] != 0.0:
        tau = (A[l][l] - A[k][k]) / (2 * A[k][l])
        if tau > 0:
            t = -tau + np.sqrt(1.0 + tau * tau)
        else:
            t = -tau - np.sqrt(1.0 + tau * tau)
        c = 1 / np.sqrt(1 + t * t)
        s = c * t
    else:
        c = 1.0
        s = 0.0
        # p.220 7 Eigensystems

    a_kk = A[k][k]
    a_ll = A[l][l]
    # changing the matrix elements with indices k and l
    A[k][k] = c * c * a_kk - 2.0 * c * s * A[k][l] + s * s * a_ll
    A[l][l] = s * s * a_kk + 2.0 * c * s * A[k][l] + c * c * a_ll
    A[k][l] = 0.0  # hard-coding of the zeros
    A[l][k] = 0.0
    # and then we change the remaining elements
    for i in range(n):
        if i != k and i != l:
            a_ik = A[i][k]
            a_il = A[i][l]
            A[i][k] = c * a_ik - s * a_il
            A[k][i] = A[i][k]
            A[i][l] = c * a_il + s * a_ik
            A[l][i] = A[i][l]

        # Finally, we compute the new eigenvectors
        r_ik = R[i][k]
        r_il = R[i][l]
        R[i][k] = c * r_ik - s * r_il
        R[i][l] = c * r_il + s * r_ik

    return


class Potential(object):

    def __init__(self, omega):
        self.omega = omega

    def __call__(self, rho):
        omega = self.omega
        return omega**2 * rho**2 + 1.0 / rho


def test_rho_max_jacobi_interactive_case(omega=0.01):   #2d
    potential = Potential(omega=omega)
    # now plot the results for the three lowest lying eigenstates
    r, eigenvectors, eigenvalues, iterations = solve_schroedinger_jacobi(
        n=320, rho_max=5, potential=potential)
    #errors = []
    #for i, trueeigenvalue in enumerate([3, 7, 11]):
        #errors.append(np.abs(eigenvalues[i] - trueeigenvalue))
        # print eigenvalues[i] - trueeigenvalue, eigenvalues[i]
    FirstEigvector = eigenvectors[:, 0]
    SecondEigvector = eigenvectors[:, 1]
    ThirdEigvector = eigenvectors[:, 2]
    plt.plot(r, FirstEigvector**2, 'b-', r, SecondEigvector **
             2, 'g-', r, ThirdEigvector**2, 'r-')
    #plt.axis([0, 4.6, 0.0, 0.025])
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$u(\rho)$')
    max_r = np.max(r)
    print omega
    #omega = np.max(errors)
    plt.suptitle(r'Normalized energy for the three lowest states interactive case.')  #as a function of various omega_r 
    plt.title(r'$\rho$ = {0:2.1f}, n={1}, omega={2:2.1g}'.format(
        max_r, len(r), omega))
    plt.savefig('eigenvector_rho{0}n{1}omega{2}.png'.format(int(max_r * 10), len(r),int(omega*100)))


def solve_schroedinger_jacobi(n=160, rho_max=5, potential=None):
    if potential is None:
        potential = lambda r: r**2
    #n = 128*4
    #n = 160
    rho_min = 0
    #rho_max = 5
    h = (rho_max - rho_min) / (n + 1)  # step_length
    rho = np.arange(1, n + 1) * h
    vi = potential(rho)

    rho = rho
    e = -np.ones(n - 1) / h**2
    d = 2 / h**2 + vi  # di
    A = np.diag(d) + np.diag(e, -1) + np.diag(e, +1)

    # Solve Schrodingers equation:
    eigenvectors, eigenvalues, iterations = jacobi_method(A)
    # self.eigenvalues, self.eigenvectors = np.linalg.eig(self.A)
    r = rho

    permute = eigenvalues.argsort()
    eigenvalues = eigenvalues[permute]
    eigenvectors = eigenvectors[:, permute]

    return r, eigenvectors, eigenvalues, iterations


def test_iterations():
        # now plot the results for the three lowest lying eigenstates
    num_iterations = []

    dims = [8, 16, 32, 64, 128, 256]
    if False:
        for n in dims:

            r, eigenvectors, eigenvalues, iterations = solve_schroedinger_jacobi(
                n=n, rho_max=5)
            num_iterations.append(iterations)
    else:
        num_iterations = [80, 374, 1623, 6741, 27070, 109974]

    step = np.linspace(0, 1.1 * dims[-1], 100)
    coeff = np.polyfit(dims, num_iterations, deg=2)
    print coeff
    plt.plot(dims, num_iterations, '.', label='Exact number of iterations')
    plt.plot(step, np.polyval(coeff, step), '-',
             label='{:0.2f}x**2+{:0.2f}x+{:0.2f}'.format(coeff[0], coeff[1], coeff[2]))
    plt.xlabel('n')
    plt.ylabel('Iterations')
    plt.title('Number of similarity transformations')
    plt.legend(loc=2)
    plt.savefig('num_iterations{0}n{1}.png'.format(dims[-1], len(dims)))
    plt.show()


def test_rho_max_jacobi():  #2b
    # now plot the results for the three lowest lying eigenstates
    r, eigenvectors, eigenvalues, iterations = solve_schroedinger_jacobi(
        n=320, rho_max=5)
    errors = []
    for i, trueeigenvalue in enumerate([3, 7, 11]):
        errors.append(np.abs(eigenvalues[i] - trueeigenvalue))
        # print eigenvalues[i] - trueeigenvalue, eigenvalues[i]
    FirstEigvector = eigenvectors[:, 0]
    SecondEigvector = eigenvectors[:, 1]
    ThirdEigvector = eigenvectors[:, 2]
    plt.plot(r, FirstEigvector**2, 'b-', r, SecondEigvector **
             2, 'g-', r, ThirdEigvector**2, 'r-')
    #plt.axis([0, 4.6, 0.0, 0.025])
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$u(\rho)$')
    max_r = np.max(r)
    max_errors = np.max(errors)
    plt.suptitle(r'Normalized energy for the three lowest states.')
    plt.title(r'$\rho$ = {0:2.1f}, n={1}, max_errors={2:2.1g}'.format(
        max_r, len(r), max_errors))
    plt.savefig('eigenvector_rho{0}n{1}.png'.format(int(max_r * 10), len(r)))
    plt.show()


def solve_schroedinger(Dim=400, RMax=10.0, RMin=0.0, lOrbital=0):
    # Get the boundary, orbital momentum and number of integration points
    # Program which solves the one-particle Schrodinger equation
    # for a potential specified in function
    # potential(). This example is for the harmonic oscillator in 3d

    # from  matplotlib import pyplot as plt
    # import numpy as np
    # Here we set up the harmonic oscillator potential

    def potential(r):
        return r * r

    # Initialize constants
    Step = RMax / (Dim + 1)
    DiagConst = 2.0 / (Step * Step)
    NondiagConst = -1.0 / (Step * Step)
    OrbitalFactor = lOrbital * (lOrbital + 1.0)

    # Calculate array of potential values
    v = np.zeros(Dim)
    r = np.linspace(RMin, RMax, Dim)
    for i in xrange(Dim):
        r[i] = RMin + (i + 1) * Step
        v[i] = potential(r[i]) + OrbitalFactor / (r[i] * r[i])

    # Setting up a tridiagonal matrix and finding eigenvectors and eigenvalues
    Hamiltonian = np.zeros((Dim, Dim))
    Hamiltonian[0, 0] = DiagConst + v[0]
    Hamiltonian[0, 1] = NondiagConst
    for i in xrange(1, Dim - 1):
        Hamiltonian[i, i - 1] = NondiagConst
        Hamiltonian[i, i] = DiagConst + v[i]
        Hamiltonian[i, i + 1] = NondiagConst
    Hamiltonian[Dim - 1, Dim - 2] = NondiagConst
    Hamiltonian[Dim - 1, Dim - 1] = DiagConst + v[Dim - 1]
    # diagonalize and obtain eigenvalues, not necessarily sorted
    EigValues, EigVectors = np.linalg.eig(Hamiltonian)
    # sort eigenvectors and eigenvalues
    permute = EigValues.argsort()
    EigValues = EigValues[permute]
    EigVectors = EigVectors[:, permute]
    return r, EigVectors, EigValues


def test_rho_max(Dim=400.0, RMax=10.0):
    r, EigVectors, EigValues = solve_schroedinger(Dim, RMax)
    # now plot the results for the three lowest lying eigenstates
    for i in xrange(3):
        print EigValues[i]
    FirstEigvector = EigVectors[:, 0]
    SecondEigvector = EigVectors[:, 1]
    ThirdEigvector = EigVectors[:, 2]
    plt.plot(r, FirstEigvector**2, 'b-', r, SecondEigvector **
             2, 'g-', r, ThirdEigvector**2, 'r-')
    plt.axis([0, 4.6, 0.0, 0.025])
    plt.xlabel(r'$r$')
    plt.ylabel(r'Radial probability $r^2|R(r)|^2$')
    plt.title(
        r'Radial probability distributions for three lowest-lying states')
    plt.savefig('eigenvector.pdf')
    plt.show()


def cpu_times_vs_dimension_plot():  #2c
    '''Jacobi loeser kvadratisk tid. 
    Det vil si tiden er bestemt av similaritetstransformasjonen for O(n**2) operasjoner.
    Eig loeser egenverdiene i lineaer tid. Det vil si at tiden er bestemt av O(n)
    '''
    cpu_times = []
    cpu_times_jacobi = []
    dims = [8, 16, 32, 64, 128]
    for n in dims:
        cpu_times.append(cpu_time(5, n))
        cpu_times_jacobi.append(cpu_time_jacobi(5, n))

    plt.plot(dims, cpu_times, label='np.linalg.eig')   #
    plt.plot(dims, cpu_times_jacobi, label='jacobi')
    plt.xlabel('dimension')
    plt.ylabel('cpu time')
    plt.title('CPU time vs dimension of matrix')
    plt.legend(loc=2)
    filename = 'cpu_time{0}n{1}.png'.format(dims[-1], len(dims))
    my_savefig(filename)
    plt.show()


if __name__ == '__main__':

    #cpu_times_vs_dimension_plot()

    # test_compare()
    # solve_poisson_with_lu(10)
    # error_test()
    # lu_test_compare()
    #     for n in [10, 100, 1000, 2000]:
    #         cpu_time_specific(10, n)
    #         cpu_time(10, n)
    #         cpu_time_lu_solve(10, n)
    #
    #     plt.show()
    # solve_schroedinger()
    # test_rho_max_jacobi()
    # test_iterations()
    
    for i, omega in enumerate([0.01, 0.5, 1, 5]):
        plt.figure(i)
        test_rho_max_jacobi_interactive_case(omega)
    plt.show()