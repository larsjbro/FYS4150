'''
Created on 9. des. 2017

@author: ljb
'''
from __future__ import division, print_function
import matplotlib.pyplot as plt
from fys4150.project5.tridiagonal_solvers import (tridiagonal_solve_specific,
                                                  tridiagonal_solve_specific_periodic,
                                                  solve_specific_periodic_with_lu,
                                                  tridiagonal_solve)
from numpy import pi
import numpy as np


def psi_periodic(x, t, n=1):
    """
    Stream function for periodic boundary conditions
    """
    return np.cos(n * 2 * pi * x + 0.5 * t / (n * pi))


def dpsi_periodic(x, t, n=1):
    """
    derivative of Stream function for periodic boundary conditions
    """
    return -n * 2 * pi * np.sin(n * 2 * pi * x + 0.5 * t / (n * pi))

def ddpsi_periodic(x, t, n=1):
    """
    double derivative of Stream function for periodic boundary conditions
    """
    return -(n * 2 * pi)**2 * np.cos(n * 2 * pi * x + 0.5 * t / (n * pi))


def zeta_periodic(x, t, n=1):
    """
    Return d^2(psi_periodic)/dx^2
    """
    return - (2 * pi * n)**2 * np.cos(n * 2 * pi * x + 0.5 * t / (n * pi))


def psi_wall(x, t, n=1):
    """
    Stream function for rigid walls
    """
    return np.sin(n * pi * x) * np.cos(n * pi * x + 0.5 * t / (n * pi))


def zeta_wall(x, t, n=1):
    """
    Return d^2(psi_wall)/dx^2
    """
    return -2 * (pi * n)**2 * np.cos(-t / (n * 2 * pi) - 2 * pi * n * x + pi / 2)


def central_diff_x(psi_n, dx):
    return (psi_n[2:] - psi_n[:-2]) / (2*dx)


def forward_diff_t(dpsi_n, zetas, dt):
    return zetas[-1] -dt * dpsi_n


def central_diff_t(dpsi_n, zetas, dt):
    return zetas[-2] - 2*dt * dpsi_n


def solve_rossby_with_walls(j_max, n_max=100, dt=0.01, method='forward'):
    """
    Return
    """
    dx = 1.0 / (j_max + 1)  # step_length
    x = np.arange(1, j_max + 1) * dx
    t = np.arange(n_max) * dt

    zetas = [zeta_wall(x, t=0)]  # Initial condition
    psis = []
    diff_x = central_diff_x
    diff_t = forward_diff_t
    if method.startswith('central'):
        # Do one iteration with forward difference before using the central method
        psi_n = tridiagonal_solve_specific(-zetas[-1] * dx ** 2)  # substitue Eq 16 into  Eq 20  and solve
        psis.append(psi_n)
        zetas.append(diff_t(diff_x(np.hstack((0, psi_n, 0)), dx), zetas, dt))
        diff_t = central_diff_t
        n_max = n_max - 1

    for n in range(n_max):
        psi_n = tridiagonal_solve_specific(-zetas[-1] * dx ** 2)  # substitue Eq 16 into  Eq 20  and solve
        psis.append(psi_n)

        zeta_np1 = diff_t(diff_x(np.hstack((0, psi_n, 0)), dx), zetas, dt)
        zetas.append(zeta_np1)

    return t, x, zetas, psis


def solve_rossby_periodic1(j_max, n_max=100, dt=0.01, method='forward'):
    """
    Return

    NB! This does not work because the matrix is singular!
    """
    dx = 1.0 / (j_max + 1)  # step_length
    x = np.arange(0, j_max + 1) * dx
    t = np.arange(n_max) * dt

    zetas = [zeta_wall(x, t=0)]  # Initial condition
    psis = []
    diff_x = central_diff_x
    diff_t = forward_diff_t
    n_start = 0
    if method.startswith('central'):
        # Do one iteration with forward difference before using the central method
        psi_n = tridiagonal_solve_specific_periodic(-zetas[-1] * dx ** 2)  # substitue Eq 16 into  Eq 20  and solve
        psis.append(psi_n)
        zetas.append(diff_t(diff_x(np.hstack((0, psi_n, 0)), dx), zetas, dt))
        diff_t = central_diff_t
        n_start = 1

    for n in range(n_start, n_max):
        psi_n = tridiagonal_solve_specific_periodic(-zetas[-1] * dx ** 2)  # substitue Eq 16 into  Eq 20  and solve
        psis.append(psi_n)

        zeta_np1 = diff_t(diff_x(np.hstack((0, psi_n, 0)), dx), zetas, dt)
        zetas.append(zeta_np1)

    return t, x, zetas, psis


def solve_rossby_periodic(j_max, zeta0, dpsi0=0, psi0=0, n_max=100, dt=0.01, method='forward'):
    """
    Return
    """
    dx = 1.0 / (j_max + 1)  # step_length
    x = np.arange(0, j_max + 1) * dx
    t = np.arange(n_max) * dt

    zetas = [zeta0(x, t=0)]  # Initial condition
    psis = []

    a = c = -np.ones(j_max+1)
    b = 2.0 * np.ones(j_max+1)
    b[0] = 1
    diff_x = central_diff_x
    diff_t = forward_diff_t
    n_start = 0
    B = B_matrix(j_max+1)
    psi0s = [psi0]
    if method.startswith('central'):
        # Do one iteration with forward difference before using the central method
        F = -zetas[-1] * dx ** 2
        F[0] = F[0]/2 - dpsi0 * dx
        F[-1] += psi0
        psi_n = tridiagonal_solve(a, b, c, F)

        psis.append(psi_n)
        dpsi_n = diff_x(np.hstack((psi_n[-1], psi_n, psi_n[0])), dx)
        ddpsi0 = (psi_n[-1] - 2*psi_n[0] + psi_n[1]) / dx**2
        # dpsi0 = dpsi_n[0] + dt * ddpsi0
        # psi0 = psi_n[0] + dt * dpsi_n[0]
        dpsi0 = dpsi_periodic(0, dt) # dpsi_n[0] + dt * ddpsi0
        psi0 =  psi_periodic(0, dt)  # psi_n[0] + dt * dpsi_n[0]
        psi0s.append(psi0)
        zetas.append(diff_t(dpsi_n, zetas, dt))
        # zetas.append(zeta0(x, t=dt))
        diff_t = central_diff_t
        n_start = 1

    for n in range(n_start, n_max):
        F = -zetas[-1] * dx ** 2
        F[0] = F[0] / 2 - dpsi0 * dx
        F[-1] += psi0
        psi_n = tridiagonal_solve(a, b, c, F)
        psi00 = psi_n[0]
#         while np.abs(psi0-psi00)>1e-4:
#             F[-1] += psi00-psi0
#             psi0 = psi00
#             psi_n = tridiagonal_solve(a, b, c, F)
#             psi00 = psi_n[0]
#             pass
        # psi_n2 = np.dot(F, B.T)
        psis.append(psi_n)
        dpsi_n = diff_x(np.hstack((psi_n[-1], psi_n, psi_n[0])), dx)
        if n>1:
            psi0_n1 = psis[-2][0]
            psi0 = psi0_n1 + 2*dt * dpsi0
        else:
            psi0 = psi0 + dt * dpsi0
        psi0s.append(psi0)
        ddpsi0 = (psi_n[-1] - 2*psi_n[0] + psi_n[1])/dx**2
        dpsi0 = dpsi_periodic(0, (n+1)*dt) # dpsi_n[0] + dt * ddpsi0
        psi0 =  psi_periodic(0, (n+1)*dt)  #
        zeta_np1 = diff_t(dpsi_n, zetas, dt)
        zetas.append(zeta_np1)

    return t, x, zetas, psis


def B_matrix(N):
    a = np.arange(N, 0, -1)
    B = [a]
    for i in range(N-1, 0, -1):
        B.append(a.clip(1, i))
    return np.array(B)


def task_5c_test():
    plot = plt.semilogy
    periodic=True
    if periodic:
        psi_fun = psi_periodic
        zeta_fun = zeta_periodic
        msg = 'Periodic'
    else:
        psi_fun = psi_wall
        zeta_fun = zeta_wall
        msg = 'Wall'

    dt = 1e-2
    nmax = int(10/dt)
    for method in ['forward', 'central']:
        plt.figure()
        if periodic:
            t, x, zetas, psis = solve_rossby_periodic(j_max=40,
                                                      zeta0=zeta_fun,
                                                      dpsi0=0,
                                                      psi0=psi_fun(x=0, t=0),
                                                      method=method,
                                                      dt=1e-2, n_max=nmax)
        else:
            t, x, zetas, psis = solve_rossby_with_walls(j_max=40, method=method, dt=dt, n_max=nmax)
#         plot(x, np.abs(psi_fun(x, t=0)-psis[0]), 'b-', label=method + ' t={}'.format(t[0]))
#         plot(x, np.abs(psi_fun(x, t=t[-1])- psis[-1]), 'r.', label=method + ' t={}'.format(t[-1]))
#         plot(x, np.abs(zeta_fun(x, t=0)-zetas[0]) + 1e-10, 'k--', label=method + ' z t={}'.format(t[0]))
#         plot(x, np.abs(zeta_fun(x, t=t[-1])- zetas[-1]), 'go', label=method + ' z t={}'.format(t[-1]))
        dx = x[1]-x[0]
        dt = t[1]-t[0]
        plt.title('{}, psi, dt={:2.1g}, dx={:2.1g}, {}'.format(msg, dt, dx, method))
        #plt.plot(x, psi_fun(x, t=0),'-')
        #plt.plot(x, psis[0], '.')
        colors = ['r', 'g', 'b']

        indices = [nmax // 3, 2 * nmax // 3, -1]
        for i, color in zip(indices, colors):
            plt.plot(x, psi_fun(x, t=t[i]),color + '-', label='t={:2.1f}'.format(t[i]))
            plt.plot(x, psis[i], color + '.', label='exact, t={:2.1f}'.format(t[i]))
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('psi')
        plt.savefig('task_5c_psi_{}_{}{}.png'.format(msg, method, nmax))
        plt.figure()

#         plt.plot(x, zeta_fun(x, t=0),'-')
#         plt.plot(x,zetas[0], '.')
        for i, color in zip(indices, colors):
            plt.plot(x, zeta_fun(x, t=t[i]),color+'-', label='t={:2.1f}'.format(t[i]))
            plt.plot(x, zetas[i], color + '.', label='exact, t={:2.1f}'.format(t[i]))
        plt.title('{}, zeta, dt={:2.1g}, dx={:2.1g} {}'.format(msg, dt, dx, method))
        plt.xlabel('x')
        plt.ylabel('zeta')
        plt.legend()
        plt.savefig('task_5c_zeta_{}_{}{}.png'.format(msg, method, nmax))
    plt.show('hold')


def task_5d_stability_and_truncation_errors():
    plot = plt.semilogy
    periodic = False
    if periodic:
        psi_fun = psi_periodic
        zeta_fun = zeta_periodic
        msg = 'Periodic'
    else:
        psi_fun = psi_wall
        zeta_fun = zeta_wall
        msg = 'Wall'

    # dt = 1e-2
    results = dict()
    dts = [0.5, 1e-1, 1e-2, 1e-3, 1e-4]
    for method in ['forward', 'central']:
        res = []
        for dt in dts:
            nmax = int(100/dt)
            print(dt, nmax)
            if periodic:
                t, x, zetas, psis = solve_rossby_periodic(j_max=40,
                                                          zeta0=zeta_fun,
                                                          dpsi0=0,
                                                          psi0=psi_fun(x=0, t=0),
                                                          method=method,
                                                          dt=1e-2, n_max=nmax)
            else:
                t, x, zetas, psis = solve_rossby_with_walls(j_max=40, method=method, dt=dt, n_max=nmax)

            indices = [nmax // 3, 2 * nmax // 3, -1]
            dx = x[1]-x[0]
            # dt = t[1]-t[0]
            for i in indices:
                ti = t[i]
                psi0 = psi_fun(x, t=ti)
                zeta0 = zeta_fun(x, t=ti)
                res.append((dx, dt, ti,
                            np.max(np.abs(psi0- psis[i])),
                            np.max(np.abs(zeta0- zetas[i]))
                            ))

        results[method] = res
    ni = len(indices)
    colors = ['r', 'g', 'b', 'm', 'c', 'y']
    for method, values in results.items():
        plt.figure()
        vals = np.array(values)

        dts = vals[:, 1].reshape(-1, ni)
        tis = vals[:, 2].reshape(-1, ni)
        error_psis = vals[:,3].reshape(-1, ni)
        error_zetas = vals[:,4].reshape(-1, ni)

        for dt, ti, error_psi, error_zeta, color,in zip(dts.T, tis.T, error_psis.T, error_zetas.T, colors):
            plt.loglog(dt, error_psi, color + '-', label='psi, t={:2.1f}'.format(np.mean(ti)))
            plt.loglog(dt, error_zeta, color + '-.', label='zeta, t={:2.1f}'.format(np.mean(ti)))
        plt.legend()
        plt.title('method={}, {}'.format(method, msg))

        plt.xlabel('dt')
        plt.ylabel('Max absolute error')
        plt.savefig('task_5d_stability_{}_{}.png'.format(msg, method))
    plt.show('hold')


if __name__ == '__main__':
    # task_5c_test()
    task_5d_stability_and_truncation_errors()
