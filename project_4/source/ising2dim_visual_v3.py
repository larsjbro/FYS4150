# coding=utf-8
# 2-dimensional ising model with visualization
# Written by Kyrre Ness Sjoebaek
from __future__ import division
import matplotlib.pyplot as plt
from numba import jit
import numpy
import numpy as np
import sys
import math
import pygame
from timeit import default_timer as timer
from scipy import integrate
from scipy import special
# Needed for visualize when using SDL
SCREEN = None
FONT = None
BLOCKSIZE = 10
T_CRITICAL = 2./ np.log(1+np.sqrt(2))



@jit(nopython=True)
def periodic(i, limit, add):
    """
    Choose correct matrix index with periodic
    boundary conditions

    Input:
    - i:     Base index
    - limit: Highest \"legal\" index
    - add:   Number to add or subtract from i
    """
    return (i + limit + add) % limit



def dump_to_terminal(spin_matrix, temp, E, M):
    # Simple terminal dump
    print "temp:", temp, "E:", E, "M:", M
    print spin_matrix


def pretty_print_to_terminal(spin_matrix, temp, E, M):
    # Pretty-print to terminal
    out = ""
    size = len(spin_matrix)
    for y in xrange(size):
        for x in xrange(size):
            if spin_matrix.item(x, y) == 1:
                out += "X"
            else:
                out += " "

        out += "\n"

    print "temp:", temp, "E:", E, "M:", M
    print out + "\n"



def display_single_pixel(spin_matrix, temp, E, M):
    # SDL single-pixel (useful for large arrays)
    size = len(spin_matrix)
    SCREEN.lock()
    for y in xrange(size):
        for x in xrange(size):
            if spin_matrix.item(x, y) == 1:
                SCREEN.set_at((x, y), (255, 255, 255))
            else:
                SCREEN.set_at((x, y), (0, 0, 0))

    SCREEN.unlock()
    pygame.display.flip()


def display_block(spin_matrix, temp, E, M):
    # SDL block (usefull for smaller arrays)
    size = len(spin_matrix)
    SCREEN.lock()
    for y in xrange(size):
        for x in xrange(size):
            if spin_matrix.item(x, y) == 1:
                rect = pygame.Rect(x * BLOCKSIZE, y * BLOCKSIZE, BLOCKSIZE, BLOCKSIZE)
                pygame.draw.rect(SCREEN, (255, 255, 255), rect)
            else:
                rect = pygame.Rect(x * BLOCKSIZE, y * BLOCKSIZE, BLOCKSIZE, BLOCKSIZE)
                pygame.draw.rect(SCREEN, (0, 0, 0), rect)

    SCREEN.unlock()
    pygame.display.flip()


def display_block_with_data(spin_matrix, E, M):
    # SDL block w/ data-display
    size = len(spin_matrix)
    SCREEN.lock()
    for y in xrange(size):
        for x in xrange(size):
            if spin_matrix.item(x, y) == 1:
                rect = pygame.Rect(x * BLOCKSIZE, y * BLOCKSIZE, BLOCKSIZE, BLOCKSIZE)
                pygame.draw.rect(SCREEN, (255, 255, 255), rect)
            else:
                rect = pygame.Rect(x * BLOCKSIZE, y * BLOCKSIZE, BLOCKSIZE, BLOCKSIZE)
                pygame.draw.rect(SCREEN, (0, 0, 0), rect)

    s = FONT.render("<E> = %5.3E; <M> = %5.3E" % E, M, False, (255, 0, 0))
    SCREEN.blit(s, (0, 0))
    SCREEN.unlock()
    pygame.display.flip()

def get_visualize_function(method):
    vis_methods = {0: dump_to_terminal,
                   1:pretty_print_to_terminal,
                   2:display_single_pixel,  # (useful for large arrays)
                   3:display_block,  # (usefull for smaller arrays)
                   4:display_block_with_data}
    def plot_nothing(spin_matrix, temp, E, M):
        pass
    return vis_methods.get(method, plot_nothing)


def visualize(spin_matrix, temp, E, M, method):
    """
    Visualize the spin matrix

    Methods:
    method = -1:No visualization (testing)
    method = 0: Just print it to the terminal
    method = 1: Pretty-print to terminal
    method = 2: SDL/pygame single-pixel
    method = 3: SDL/pygame rectangle
    """
    get_visualize_function(method)(spin_matrix, temp, E, M)



@jit(nopython=True)
def metropolis(E, M, w, size, spin_matrix):
    # Metropolis
    # Loop over all spins, pick a random spin each time
    number_of_accepted_configurations = 0
    for s in xrange(size**2):
        x = int(numpy.random.random() * size)
        y = int(numpy.random.random() * size)
        deltaE = 2 * spin_matrix[x, y] * (spin_matrix[periodic(x, size, -1), y]
                                          + spin_matrix[periodic(x, size, 1), y]
                                          + spin_matrix[x, periodic(y, size, -1)]
                                          + spin_matrix[x, periodic(y, size, 1)])

        accept = numpy.random.random() <= w[deltaE + 8]
        if accept:
            spin_matrix[x, y] *= -1
            M += 2 * spin_matrix[x, y]
            E += deltaE
            number_of_accepted_configurations += 1
    return E, M, number_of_accepted_configurations


@jit(nopython=True)
def _compute_initial_energy(spin_matrix, size):
# Calculate initial energy
    E = 0
    for j in xrange(size):
        for i in xrange(size):
            E -= spin_matrix[i, j] * (spin_matrix[periodic(i, size, -1), j]
                                      + spin_matrix[i, periodic(j, size, 1)])
    return E


def monteCarlo(temp, size, trials, visualizer=None, spin_matrix='ordered'):
    """
    Calculate the energy and magnetization
    (\"straight\" and squared) for a given temperature

    Input:
    - temp:   Temperature to calculate for units Kb Kelvin / J
    - size:   dimension of square matrix
    - trials: Monte-carlo trials (how many times do we
                                  flip the matrix?)
    - visual_method: What method should we use to visualize?

    Output:
    - E_av:       Energy of matrix averaged over trials, normalized to spins**2
    - E_variance: Variance of energy, same normalization * temp**2
    - M_av:       Magnetic field of matrix, averaged over trials, normalized to spins**2
    - M_variance: Variance of magnetic field, same normalization * temp
    - Mabs:       Absolute value of magnetic field, averaged over trials
    - Mabs_variance
    - num_accepted_configs
    """
    if visualizer is None:
        visualizer = get_visualize_function(method=None)  # No visualization
    # Setup spin matrix, initialize to ground state
    if spin_matrix == 'ordered':
        spin_matrix = numpy.zeros((size, size), numpy.int8) + 1
    elif spin_matrix == 'random':
        spin_matrix = np.array(numpy.random.random(size=(size, size))>0.5, dtype=numpy.int8)
    else:
        raise NotImplementedError('method')

    # Create and initialize variables
    E_av = E2_av = M_av = M2_av = Mabs_av = 0.0

    # Setup array for possible energy changes
    w = numpy.zeros(17, dtype=float)
    for de in xrange(-8, 9, 4):  # include +8
        w[de + 8] = math.exp(-de / temp)

    # Calculate initial magnetization:
    M = spin_matrix.sum()
    E = _compute_initial_energy(spin_matrix, size)

    total_accepted_configs = 0
    # Start metropolis MonteCarlo computation
    for i in xrange(trials):
        E, M, num_accepted_configs = metropolis(E, M, w, size, spin_matrix)
        # Update expectation values
        total_accepted_configs += num_accepted_configs
        E_av += E
        E2_av += E**2
        M_av += M
        M2_av += M**2
        Mabs_av += int(math.fabs(M))

        visualizer(spin_matrix, temp, E / float(size**2), M / float(size**2))

    # Normalize average values
    E_av /= float(trials)
    E2_av /= float(trials)
    M_av /= float(trials)
    M2_av /= float(trials)
    Mabs_av /= float(trials)
    # Calculate variance and normalize to per-point and temp
    E_variance = (E2_av - E_av * E_av) / float(size * size * temp * temp)
    M_variance = (M2_av - M_av * M_av) / float(size * size * temp)
    Mabs_variance = (M2_av - Mabs_av * Mabs_av) / float(size * size * temp)

    # Normalize returned averages to per-point
    E_av /= float(size * size)
    M_av /= float(size * size)
    Mabs_av /= float(size * size)

    return E_av, E_variance, M_av, M_variance, Mabs_av, Mabs_variance, total_accepted_configs


def initialize_pygame(size, method):
    global SCREEN, FONT
# Initialize pygame
    if method == 2 or method == 3 or method == 4:
        pygame.init()
        if method == 2:
            SCREEN = pygame.display.set_mode((size, size))
        elif method == 3:
            SCREEN = pygame.display.set_mode((size * 10, size * 10))
        elif method == 4:
            SCREEN = pygame.display.set_mode((size * 10, size * 10))
            FONT = pygame.font.Font(None, 12)



def partition2(T):
    '''
    Return the partition2 function for 2x2 lattice

    Parameters:
    -----------
    T: arraylike
        normalized temperature in units of kb*K/J, kb = boltzmann's constant, K = Kelvin,
        J is the coupling constant expressing the strength of the interaction between neighbouring spins
    '''
    z = 12+4*np.cosh(8.0/T)
    return z


def partition(T, size=2):
    '''
    Return the partition function for size x size lattice

    Parameters:
    -----------
    T: arraylike
        normalized temperature in units of kb*K/J, kb = boltzmann's constant, K = Kelvin,
        J is the coupling constant expressing the strength of the interaction between neighbouring spins
    '''
    kappa = 2*np.sinh(2./T)/np.cosh(2./T)**2
    N = size**2

    k1 = special.ellipk(kappa**2)


def energy_mean_asymptotic(T):
    '''
    Return mean energy for size x size lattice normalized by spins**2 * size **2

    Parameters:
    -----------
    T: arraylike
        normalized temperature in units of kb*K/J, kb = boltzmann's constant, K = Kelvin,
        J is the coupling constant expressing the strength of the interaction between neighbouring spins

    Output:
    - E_av:   mean of energy, normalized to spins**2

    page 428 in lecture notes
    '''
    denominator = np.tanh(2.0/T)
    kappa = 2*denominator/np.cosh(2./T)

    k1 = special.ellipk(kappa**2)

#     k = 1./np.sinh(2.0/T)**2
#     def integrand(theta):
#         return 1./np.sqrt(1-4.*k*(np.sin(theta)/(1+k))**2)
#     k11, abserr = integrate.quad(integrand, 0, np.pi/2, epsabs=1e-3, epsrel=1e-3)


    return  -(1+ 2/np.pi *(2*denominator**2-1)*k1)/denominator


def energy_mean2(T):
    '''
    Return mean energy for 2 x 2 lattice normalized by spins**2 * 4

    Parameters:
    -----------
    T: arraylike
        normalized temperature in units of kb*K/J, kb = boltzmann's constant, K = Kelvin,
        J is the coupling constant expressing the strength of the interaction between neighbouring spins

    Output:
    - E_av:   mean of energy, normalized to spins**2

    '''
    size = 2
    return -8*np.sinh(8.0/T)/(np.cosh(8.0/T)+3)/size**2


def energy_variance2(T):
    '''
    Return variance of energy for 2 x 2 lattice normalized by spins**2 * 4 * T**2

    Output:
    -E_variance: Variance of energy, same normalization * temp**2 per cell
    '''
    size = 2
    return 64.0*(1.+3.*np.cosh(8.0/T))/(np.cosh(8.0/T)+3)**2 / size**2 / T**2


def energy_variance(T):
    '''
    Return variance of energy for size x size lattice normalized by spins**2 * size**2 * T**2

    Output:
    -E_variance: Variance of energy, same normalization * temp**2 per cell
    '''
    tanh2 = np.tanh(2./T)**2
    kappa = 2*np.sinh(2./T)/np.cosh(2./T)**2
    # N = size**2

    k1 = special.ellipk(kappa**2)
    k2 = special.ellipe(kappa**2)

    return 4/np.pi * (k1-k2 -(1-tanh2)*(np.pi/2+(2*tanh2-1)*k1))/tanh2/T**2


def energy_mean_and_variance2(temperature):
    return energy_mean2(temperature), energy_variance2(temperature)


def specific_heat2(T):
    return energy_variance2(T)


def magnetization_spontaneous_asymptotic(T):
    """ Return spontaneous magnetization for size x size lattice normalized by spins**2 * size**2
    for T < Tc= 2.269
    """
    tanh2 = np.tanh(1./T)**2
    return (1 - (1-tanh2)**4/(16*tanh2**2))**(1./8)  # pp 429
    # return (1 - 1./np.sinh(2./T)**4)**(1./8)


def magnetization_mean2(T):
    '''
    Output:
    - M_av:       Magnetic field of matrix, averaged over trials, normalized to spins**2 per cell
    '''

    return np.where(T>T_CRITICAL, 0, magnetization_spontaneous_asymptotic(T))


def magnetization_variance2(T):
    """Return variance of magnetization for 2 x 2 lattice normalized by spins**2 * 4 * T"""

    size = 2 * 2
    denominator = np.cosh(8./T) + 3
    mean = magnetization_mean2(T) * size
    sigma = 8.0 * (np.exp(8./T) + 1) / denominator - mean**2
    return sigma / size / T


def magnetization_mean_and_variance2(T):
    """Return normalized mean and variance for the moments of a 2 X 2 ising model."""
    return magnetization_mean2(T), magnetization_variance2(T)


def susceptibility2(T):
    '''
    Output:
    - M_variance: Variance of magnetic field, same normalization * temp
    '''
    return magnetization_variance2(T)


def magnetization_abs_mean2(T):
    '''
    Lattice 2x2
    Output:
    - Mabs:       Absolute value of magnetic field, averaged over trials per cell
    '''
    size = 2
    return (2*np.exp(8.0/T)+4)/(np.cosh(8.0/T)+3)/size**2


def magnetization_abs_mean_and_variance2(temperature):
    """Return normalized mean and variance for the moments of a 2 X 2 ising model."""
    beta = 1. / temperature
    size = 2
    denominator = (np.cosh(8 * beta) + 3)
    mean = 2 * (np.exp(8 * beta) + 2) / denominator
    sigma = (8 * (np.exp(8 * beta) + 1) / denominator - mean**2) / temperature
    return mean / size**2, sigma / size**2


def read_input():
    """
    method = -1:No visualization (testing)
    method = 0: Just print it to the terminal
    method = 1: Pretty-print to terminal
    method = 2: SDL/pygame single-pixel
    method = 3: SDL/pygame rectangle
    """
    if len(sys.argv) == 5:
        size = int(sys.argv[1])
        trials = int(sys.argv[2])
        temperature = float(sys.argv[3])
        method = int(sys.argv[4])
    else:
        print "Usage: python", sys.argv[0],\
              "lattice_size trials temp method"
        sys.exit(0)

    if method > 4:
        print "method < 3!"
        sys.exit(0)
    return size, trials, temperature, method


def plot_abs_error_size2(trial_sizes, data, names, truths, temperature):
    for i, truth in enumerate(truths):
        name = names[i]
        print i
        plt.loglog(trial_sizes, np.abs(data[:,i] - truth), label=name)

    plt.title('T={} $[k_b K / J]$'.format(temperature))
    plt.ylabel('Absolute error')
    plt.xlabel('Number of trials')
    plt.legend(framealpha=0.2)


def plot_rel_error_size2(trial_sizes, data, names, truths, temperature):
    for i, truth in enumerate(truths):
        name = names[i]
        print i
        if truth == 0:
            scale = 1e-3
        else:
            scale = np.abs(truth) + 1e-16
        plt.loglog(trial_sizes, np.abs(data[:,i] - truth)/scale, label=name)


    plt.title('T={} $[k_b K / J]$'.format(temperature))
    plt.ylabel('Relative error')
    plt.xlabel('Number of trials')
    plt.legend(framealpha=0.2)


def compute_monte_carlo(temperature, size, trial_sizes, spin_matrix='ordered'):
    data = []
    for trials in trial_sizes:
        print trials
        t0 = timer()
        data.append(monteCarlo(temperature, size, trials, spin_matrix=spin_matrix))
        print 'elapsed time: {} seconds'.format(timer() - t0)

    data = np.array(data)
    names = ['Average Energy per spin $E/N$',
             'Specific Heat per spin $C_V/N$',
             'Average Magnetization per spin $M/N$',
             'Susceptibility per spin $var(M)/N$',
             'Average |Magnetization| per spin $|M|/N$',
             'Variance of |Magnetization| per spin $var(|M|)/N$',
             'Number of accepted configurations']
    return data, names


def plot_mean_energy_and_magnetization(trial_sizes, data, names, temperature, spin_matrix='ordered'):


    ids = [0, 4]
    for i in ids:
        name = names[i]
        print i
        plt.semilogx(trial_sizes, data[:,i], label=name)


    plt.title('T={} $[k_b K / J]$, {} spin matrix'.format(temperature, spin_matrix))

    plt.ylabel('E or |M|')
    plt.xlabel('Number of trials')
    plt.legend()


def plot_total_number_of_accepted_configs(trial_sizes, data, names, temperature, spin_matrix='ordered'):
    i = 6
    # for i in ids:
    name = names[i]
    print i
    plt.loglog(trial_sizes, data[:,i], label=name)

    x = np.log(trial_sizes)
    y = np.log(data[:, i])
    mask = np.isfinite(y)
    p = np.polyfit(x[mask], y[mask], deg=1)
    sgn = '+' if p[1] > 0 else '-'
    label_p = 'exp({:2.1f} {} {:2.1f} ln(x))'.format(p[0], sgn, abs(p[1]))
    plt.loglog(trial_sizes, np.exp(np.polyval(p, x)), label=label_p)

    plt.title('T={} $[k_b K / J]$, {} spin matrix'.format(temperature, spin_matrix))

    plt.ylabel('')
    plt.xlabel('Number of trials')
    plt.legend()


def main(size, trials, temperature, method):
    initialize_pygame(size, method)
    visualizer = get_visualize_function(method)
    (E_av, E_variance, M_av, M_variance, Mabs_av) = monteCarlo(temperature, size, trials, visualizer)
    print "T=%15.8E E[E]=%15.8E Var[E]=%15.8E E[M]=%15.8E Var[M]=%15.8E E[|M|]= %15.8E\n" % (temperature, E_av, E_variance, M_av, M_variance, Mabs_av)

    pygame.quit()


def task_b(temperatures=(1, 2.4)):
    size = 2
    trial_sizes = 10**np.arange(1, 6)
    for temperature in temperatures:
        data, names = compute_monte_carlo(temperature, size, trial_sizes)

        truths = (energy_mean_and_variance2(temperature)
                  + magnetization_mean_and_variance2(temperature)
                  + magnetization_abs_mean_and_variance2(temperature))

        plt.figure()
        plot_abs_error_size2(trial_sizes, data, names, truths, temperature)
        plt.savefig('task_b_abserr_T{}_size{}.png'.format(temperature, 2))
        plt.figure()
        plot_rel_error_size2(trial_sizes, data, names, truths, temperature)
        plt.savefig('task_b_relerr_T{}_size{}.png'.format(temperature, 2))
    # plt.show('hold')


def task_c(temperatures=(1,)):
    trial_sizes = [10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000] # 10**np.arange(1, 5)
    for temperature in temperatures:
        print temperature
        size = 20
        for spin_matrix in ['ordered', 'random']:
            data, names = compute_monte_carlo(temperature, size, trial_sizes, spin_matrix)

            plt.figure()
            plot_mean_energy_and_magnetization(trial_sizes, data, names, temperature, spin_matrix)
            plt.savefig('task_c_T{}_size{}_{}.png'.format(temperature, 20, spin_matrix))

            plt.figure()
            plot_total_number_of_accepted_configs(trial_sizes, data, names, temperature, spin_matrix)
            plt.savefig('task_c_accepted_T{}_size{}_{}.png'.format(temperature, 20, spin_matrix))

    # plt.show('hold')


if __name__ == '__main__':
#     T=2.4
#
#     print energy_mean2(T), energy_mean_asymptotic(T), magnetization_spontaneous_asymptotic(T), magnetization_mean2(T)
#     print energy_variance2(T)/energy_variance(T)
    # task_b(temperatures=(1,2.4))
    # task_c(temperatures=(1, 2.0, 2.4, 5, 10))
    task_c(temperatures=(1, 2.4))
    plt.show('hold')
    # Main program

#     # Get input
#     # size, trials, temperature, method = read_input()
#     size = 2
#     trials = 1000
#     temperature = 4
#     method = -1
#
#     main(size, trials, temperature, method)
#     print energy_mean_and_variance2(temperature)
#     print magnetization_mean_and_variance2(temperature)
#     print magnetization_abs_mean_and_variance2(temperature)


