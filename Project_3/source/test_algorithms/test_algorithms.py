'''
Created on 26. okt. 2017

@author: LJB
'''
from FYS4150.FYS4150.Project_3.source.solar_systems import EarthSunSystem, EarthSunSystem2
from FYS4150.FYS4150.Project_3.source.ode_solvers import ForwardEuler, CentralEuler, VelocityVerlet
from FYS4150.FYS4150.Project_3.source.utilities import plot_earth_orbit, solve_and_plot_earth_orbit, solve_timer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import OrderedDict


def stability_test_generator(spacings):
    if spacings is None:
        spacings = [1.0 / 2 ** k for k in range(4, 9)]
    x0, y0, x0_der, y0_der = initial_conditions_earth_sun_system()

    for solver in [ForwardEuler, CentralEuler]:
        for dt in spacings:
            yield EarthSunSystem([x0, y0, x0_der, y0_der], dt=dt), solver

    for dt in spacings:
        yield EarthSunSystem2([x0, y0], dt=dt, v0=[x0_der, y0_der]), VelocityVerlet


def test_stability_of_all_algorithms(spacings=None):  # 3c
    # examine stability of solution for different discretizations.

    results = []
    for problem, solver in stability_test_generator(spacings):
        plt.figure()
        results.append(solve_and_plot_earth_orbit(problem, solver))
    table = pd.DataFrame(results)
    table.to_csv('test_algorithms.csv')
    plt.show('hold')


def test_kinetic_potential_conserved():
    spacings = [1.0 / 2 ** k for k in range(8, 9)]

    for problem, solverclass in stability_test_generator(spacings):
        U, t, elapsed_time, solver = solve_timer(problem, solverclass)
        classname = solver.name
        x, y = U[:, :2].T
        if U.shape[1] > 2:
            u, v = U[:, 2:].T
        else:
            u, v = solver.v[:, :2].T
        pos = np.vstack((x, y, np.zeros_like(x))).T
        W = np.vstack((u, v, np.zeros_like(x))).T
        L = np.linalg.norm(np.cross(pos, W, axis=1), axis=1)
        r = np.sqrt((pos**2).sum(axis=1))

        # plt.plot(np.arange(len(L)), L)
        # plt.show()
        # test conserved quantities: kinetic and potential energy, angular momentum. u(x), v(y)
        vtot2 = u**2 + v**2
        msg = '{} max diff(vtot2) = {}, std(vtot2) = {}, std(L) = {}, std(1/r) = {}'
        print msg.format(classname, np.max(np.abs(np.diff(vtot2))), np.std(vtot2), np.std(L), np.std(1. / r))


def initial_conditions_earth_sun_system():
    x0 = 1  # AU
    y0 = 0
    x0_der = 0
# solving for v: v**2*r=g*m=4*pi**2, r=1AU at initial position
    y0_der = 2 * np.pi
    return x0, y0, x0_der, y0_der


def escape_velocity_problems_generator(scalings=None, betas=(2,)):
    if scalings is None:
        scalings = 2**np.r_[0, 0.1, 0.2, .4, .8]
    dt = 1. / 2**9
    x0, y0, x0_der, y0_der = initial_conditions_earth_sun_system()
    for beta in betas:
        for scale in scalings:
            yield EarthSunSystem2([x0, y0], dt=dt, v0=[x0_der, y0_der * scale], beta=beta), VelocityVerlet


def test_escape_velocity():  # 3c
    '''
    First figure out escape velocity by trial and error.
    Replace gravitational force with a dependence on 1/(r**beta), beta element of [2,3]. What happens when beta goes towards 3?
    '''
    test_scaling = False
    if test_scaling:
        scalings = [1, 1.2, 1.4, 1.5]
        betas = [2, ]
    else:  # test_beta
        scalings = [1, ]
        betas = [2, 2.3, 2.7, 3.0]

    v00 = 2 * np.pi
    for problem, solverclass in escape_velocity_problems_generator(scalings, betas):
        U, t, elapsed_time, solver = solve_timer(problem, solverclass)
        classname = solver.name
        x, y = U[:, :2].T
        if U.shape[1] > 2:
            u, v = U[:, 2:].T
        else:
            u, v = solver.v[:, :2].T

        v0 = problem.v0[-1]
        beta = int(np.round(100 * problem.beta))
        k = int(np.round(np.abs(np.log2(problem.dt))))
        filename = 'escape_velocity_test_k{}beta{}v0{}{}.png'.format(
            k, beta, 2 * v0 / v00, classname)
        print(filename)
        plt.figure()
        plot_earth_orbit(x, y)
        plt.savefig(filename)
        plt.suptitle('Escape velocity')
        plt.title('dt={}[year], beta={}, v0={}$\pi$[AU/yr]'.format(problem.dt,
                                                                   problem.beta, 2 * v0 / v00))
        plt.legend()         #
        plt.savefig(filename)
    plt.show('hold')


def analyze_test_algorithm_data():
    df = pd.read_csv('test_algorithms.csv')
    # table = table.set_index(['method','elapsed_time'])
    # df.pivot_table(df, index=['subject','date'],columns='strength')
    d = OrderedDict(k=list(range(4, 9)))
    for name in sorted(set(df.method)):
        d[name] = df.elapsed_time.values[df.method == name]
    table = pd.DataFrame(d)

    print table.to_latex()
    # print table


if __name__ == '__main__':
    test_stability_of_all_algorithms()
    #analyze_test_algorithm_data()
    # test_kinetic_potential_conserved()
    # test_escape_velocity()
