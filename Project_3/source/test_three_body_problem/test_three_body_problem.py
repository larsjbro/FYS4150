'''
Created on 27. okt. 2017

@author: ljb
'''
from __future__ import division
from FYS4150.FYS4150.Project_3.source.solar_systems import EarthSunJupiterSystem, SolarSystem
from FYS4150.FYS4150.Project_3.source.ode_solvers import VelocityVerlet
from FYS4150.FYS4150.Project_3.source.utilities import solve_and_plot_earth_and_jupiter_orbit, solve_timer,\
    solve_and_animate_earth_and_jupiter_orbit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def initial_conditions_earth_jupiter_sun_system():
    r"""
    Start positions collected from https://ssd.jpl.nasa.gov/horizons.cgi

    Jupiter start position
    2458053.500000000 = A.D. 2017-Oct-27 00:00:00.0000 TDB
    X =-4.544631393478457E+00 Y =-2.980888528535559E+00 Z = 1.140141125335228E-01
    VX= 4.050951012061073E-03 VY=-5.951569075599745E-03 VZ=-6.588031127033595E-05
    LT= 3.139692571486596E-02 RG= 5.436209170000878E+00 RR=-1.244665947680287E-04

    Earth start position
    2458053.500000000 = A.D. 2017-Oct-27 00:00:00.0000 TDB
    X = 8.307575140688143E-01 Y = 5.546449644310195E-01 Z =-1.569494708463199E-04
    VX=-9.791937393985607E-03 VY= 1.428201625774134E-02 VZ= 1.082897034006722E-07
    LT= 5.769130276129173E-03 RG= 9.988939425102245E-01 RR=-2.135133430950652E-04
    """
    # Jupiter start position
    # 2458053.500000000 = A.D. 2017-Oct-27 00:00:00.0000 TDB
    Xj = [-4.544631393478457E+00, -2.980888528535559E+00, 1.140141125335228E-01]
    Vj = [4.050951012061073E-03, -5.951569075599745E-03, -6.588031127033595E-05]
    # LT= 3.139692571486596E-02 RG= 5.436209170000878E+00 RR=-1.244665947680287E-04

    # Earth start position
    # 2458053.500000000 = A.D. 2017-Oct-27 00:00:00.0000 TDB
    X = [8.307575140688143E-01, 5.546449644310195E-01, -1.569494708463199E-04]
    V = [-9.791937393985607E-03, 1.428201625774134E-02, 1.082897034006722E-07]
    # LT= 5.769130276129173E-03 RG= 9.988939425102245E-01 RR=-2.135133430950652E-04

    X_all = X + Xj
    V_all = np.array(V + Vj) * 365  # AU / year
    return X_all, V_all.tolist()


def stability_problems_generator(spacings):
    if spacings is None:
        spacings = [1.0 / 2 ** k for k in range(4, 9)]
    x0, v0 = initial_conditions_earth_jupiter_sun_system()
    for dt in spacings:
        yield EarthSunJupiterSystem(x0, dt=dt, v0=v0)
        #yield SolarSystem(x0 + [0,0,0], gm=(3.0e-6, 0.95e-3, 1), dt=dt, v0=v0 + [0,0,0])


def test_stability(spacings=None):  # 3e
    # examine stability of solution for different discretizations.

    results = []
    for problem in stability_problems_generator(spacings):
        results.append(solve_and_plot_earth_and_jupiter_orbit(problem, VelocityVerlet))
    table = pd.DataFrame(results)
    table.to_csv('test_stability_problem.csv')
    plt.show('hold')


def jupiter_mass_problems_generator(scalings, spacings):

    x0, v0 = initial_conditions_earth_jupiter_sun_system()
    for scale in scalings:
        plt.figure()
        for dt in spacings:
            yield EarthSunJupiterSystem(x0, gm=(3.0e-6, 0.95e-3 * scale), dt=dt, v0=v0)
            # yield SolarSystem(x0+[0,0,0], gm=(3.0e-6, 0.95e-3*scale, 1), dt=dt, v0=v0 +[0,0,0])


def test_larger_jupiter_mass(scalings=None, spacings=None):  # 3e
    # examine stability of solution for different discretizations.
    if scalings is None:
        scalings = [10.5, 1050]
    if spacings is None:
        spacings = [1. / 2**k for k in range(8, 9)]

    results = []
    for problem in jupiter_mass_problems_generator(scalings, spacings):
        results.append(solve_and_plot_earth_and_jupiter_orbit(problem, VelocityVerlet))
    table = pd.DataFrame(results)
    table.to_csv('test_larger_jupiter_mass_problem.csv')
    plt.show('hold')


def test_animation_of_larger_jupiter_mass(scalings=None, spacings=None):  # 3e
    # examine stability of solution for different discretizations.
    if scalings is None:
        scalings = [10.5, 1050]
    if spacings is None:
        spacings = [1. / 2**k for k in range(8, 9)]

    results = []
    for problem in jupiter_mass_problems_generator(scalings, spacings):
        results.append(solve_and_animate_earth_and_jupiter_orbit(problem, VelocityVerlet))
    table = pd.DataFrame(results)
    table.to_csv('test_animation_of_larger_jupiter_mass_problem.csv')
    # plt.show('hold')


if __name__ == '__main__':
    #test_stability()
    test_larger_jupiter_mass()
    #test_animation_of_larger_jupiter_mass()
