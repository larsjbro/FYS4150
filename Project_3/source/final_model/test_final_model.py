'''
Created on 29. okt. 2017

@author: ljb
'''
from __future__ import division

from fys4150.project3.solar_systems import SolarSystem
from fys4150.project3.ode_solvers import VelocityVerlet
from fys4150.project3.utilities import (solve_and_plot_earth_and_jupiter_orbit,
                                        solve_timer,
                                        solve_and_animate_earth_and_jupiter_orbit,
                                        solve_and_plot_earth_jupiter_and_sun_orbit,
                                        solve_and_plot_all_planet_orbits)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import OrderedDict


r"""
Start positions collected from https://ssd.jpl.nasa.gov/horizons.cgi

Coordinate system description:

Ecliptic and Mean Equinox of Reference Epoch

Reference epoch: J2000.0
XY-plane: plane of the Earth's orbit at the reference epoch
         Note: obliquity of 84381.448 arcseconds wrt ICRF equator (IAU76)
X-axis  : out along ascending node of instantaneous plane of the Earth's
         orbit and the Earth's mean equator at the reference epoch
Z-axis  : perpendicular to the xy-plane in the directional (+ or -) sense
         of Earth's north pole at the reference epoch.

Symbol meaning [1 au= 149597870.700 km, 1 day= 86400.0 s]:

JDTDB    Julian Day Number, Barycentric Dynamical Time
 X      X-component of position vector (au)
 Y      Y-component of position vector (au)
 Z      Z-component of position vector (au)
 VX     X-component of velocity vector (au/day)
 VY     Y-component of velocity vector (au/day)
 VZ     Z-component of velocity vector (au/day)
 LT     One-way down-leg Newtonian light-time (day)
 RG     Range; distance from coordinate center (au)
 RR     Range-rate; radial velocity wrt coord. center (au/d

Earth start position
2458053.500000000 = A.D. 2017-Oct-27 00:00:00.0000 TDB
X = 8.307575140688143E-01 Y = 5.546449644310195E-01 Z =-1.569494708463199E-04
VX=-9.791937393985607E-03 VY= 1.428201625774134E-02 VZ= 1.082897034006722E-07
LT= 5.769130276129173E-03 RG= 9.988939425102245E-01 RR=-2.135133430950652E-04

Jupiter start position

2458053.500000000 = A.D. 2017-Oct-27 00:00:00.0000 TDB
X =-4.544631393478457E+00 Y =-2.980888528535559E+00 Z = 1.140141125335228E-01
VX= 4.050951012061073E-03 VY=-5.951569075599745E-03 VZ=-6.588031127033595E-05
LT= 3.139692571486596E-02 RG= 5.436209170000878E+00 RR=-1.244665947680287E-04

Mars
2458053.500000000 = A.D. 2017-Oct-27 00:00:00.0000 TDB
X =-1.600636804697038E+00 Y = 4.512663808098941E-01 Z = 4.854202215913982E-02
VX=-3.228847514312681E-03 VY=-1.228157479108209E-02 VZ=-1.782435573599619E-04
LT= 9.608969664097331E-03 RG= 1.663741522868051E+00 RR=-2.300248782067990E-04

Venus
2458053.500000000 = A.D. 2017-Oct-27 00:00:00.0000 TDB
X =-7.049960476048377E-01 Y = 1.312910376443176E-01 Z = 4.240058086399950E-02
VX=-3.646727612165235E-03 VY=-1.999997620788355E-02 VZ=-6.415907193191676E-05
LT= 4.148955550429753E-03 RG= 7.183693847609103E-01 RR=-8.019475643934765E-05

Saturn
2458053.500000000 = A.D. 2017-Oct-27 00:00:00.0000 TDB
X =-3.000365464147523E-01 Y =-1.005121487413227E+01 Z = 1.867018683325358E-01
VX= 5.269787506575000E-03 VY=-1.837774500581324E-04 VZ=-2.068472851032913E-04
LT= 5.808684312418366E-02 RG= 1.005742511594300E+01 RR= 2.261404072431979E-05

Marcury
2458053.500000000 = A.D. 2017-Oct-27 00:00:00.0000 TDB
X =-1.537256803720000E-01 Y =-4.326509049973161E-01 Z =-2.165294807336455E-02
VX= 2.084982792680136E-02 VY=-8.026395475494962E-03 VZ=-2.569426424260501E-03
LT= 2.654774284805334E-03 RG= 4.596599183756387E-01 RR= 7.029304202490583E-04

Uranus
2458053.500000000 = A.D. 2017-Oct-27 00:00:00.0000 TDB
X = 1.784192250600989E+01 Y = 8.843248837812927E+00 Z =-1.983010713614406E-01
VX=-1.775564692276439E-03 VY= 3.340680253439256E-03 VZ= 3.546912825111671E-05
LT= 1.150150026748974E-01 RG= 1.991423039017187E+01 RR=-1.076644389161439E-04

Neptune
2458053.500000000 = A.D. 2017-Oct-27 00:00:00.0000 TDB
X = 2.862286355822386E+01 Y =-8.791151564880529E+00 Z =-4.786054033036537E-01
VX= 9.010839253968958E-04 VY= 3.019851091079401E-03 VZ=-8.280715265026730E-05
LT= 1.729554859534162E-01 RG= 2.994631408439897E+01 RR=-2.393397869247329E-05
"""

PLANETS = OrderedDict(Earth=dict(X=[8.307575140688143E-01, 5.546449644310195E-01, -1.569494708463199E-04],
                                 V=[-9.791937393985607E-03, 1.428201625774134E-02, 1.082897034006722E-07],
                                 mass=6e+24))


PLANETS['Jupiter'] = dict(X=[-4.544631393478457E+00, -2.980888528535559E+00, 1.140141125335228E-01],
                          V=[4.050951012061073E-03, -5.951569075599745E-03, -6.588031127033595E-05],
                          mass=1.9e+27)

PLANETS['Mars'] = dict(X=[-1.600636804697038E+00, 4.512663808098941E-01, 4.854202215913982E-02],
                       V=[-3.228847514312681E-03, -1.228157479108209E-02, -1.782435573599619E-04],
                       mass=6.6e+23)

PLANETS['Venus'] = dict(X=[-7.049960476048377E-01, 1.312910376443176E-01, 4.240058086399950E-02],
                         V=[-3.646727612165235E-03, -1.999997620788355E-02, -6.415907193191676E-05],
                         mass=4.9e+24)

PLANETS['Saturn'] = dict(X=[-3.000365464147523E-01, -1.005121487413227E+01, 1.867018683325358E-01],
                          V=[5.269787506575000E-03, -1.837774500581324E-04, -2.068472851032913E-04],
                          mass=5.5e+26)
PLANETS['Marcury'] = dict(X=[-1.537256803720000E-01, -4.326509049973161E-01, -2.165294807336455E-02],
                           V=[2.084982792680136E-02, -8.026395475494962E-03, -2.569426424260501E-03],
                           mass=3.3e+23)
PLANETS['Uranus'] = dict(X=[1.784192250600989E+01, 8.843248837812927E+00, -1.983010713614406E-01],
                          V=[-1.775564692276439E-03, 3.340680253439256E-03, 3.546912825111671E-05],
                          mass=8.8e+25)

PLANETS['Neptune'] = dict(X=[2.862286355822386E+01, -8.791151564880529E+00, -4.786054033036537E-01],
                           V=[9.010839253968958E-04, 3.019851091079401E-03, -8.280715265026730E-05],
                           mass=1.03e+26)
PLANETS['Sun'] = dict(X=[0, 0, 0],
                      V=[0, 0, 0],  # Unknown start velocity
                      mass=2e+30)




def initial_conditions_solar_system():

    all_masses = np.array([PLANETS[name]['mass'] for name in PLANETS]) / PLANETS['Sun']['mass']
    all_X = np.vstack([PLANETS[name]['X'] for name in PLANETS])
    all_V = np.vstack([PLANETS[name]['V'] for name in PLANETS]) * 365  # AU / year


    center_of_mass = np.zeros(3)
    M = sum(all_masses)
    for i, mi in enumerate(all_masses):
        center_of_mass += mi * all_X[i, :]
    center_of_mass = center_of_mass / M
    print center_of_mass
    # center_of_mass = np.sum(all_masses[:, None] * all_X, axis=0)/M
    # print center_of_mass

    all_X = all_X - center_of_mass[None, :]

    print np.sum(all_masses[:, None] * all_X, axis=0) / M   # should be zero if all is correct!

    # all_V = np.vstack((V, Vj, V_sun)) * 365  # AU / year

    # Make sure the total momentum is zero
    all_V[-1] = -np.sum(all_masses[:-1, None] * all_V[:-1, :], axis=0)

    print np.sum(all_masses[:, None] * all_V, axis=0)  # should be zero if all is correct!

    return all_X.ravel(), all_V.ravel().tolist(), all_masses



def stability_problems_generator(spacings):
    if spacings is None:
        spacings = [1.0 / 2 ** k for k in range(4, 9)]
    x0, v0, all_masses = initial_conditions_solar_system()
    for dt in spacings:
        yield SolarSystem(x0, gm=all_masses, dt=dt, v0=v0)


def test_stability(spacings=None):  # 3e
    # examine stability of solution for different discretizations.

    results = []
    for problem in stability_problems_generator(spacings):
        # results.append(solve_and_plot_earth_and_jupiter_orbit(problem, VelocityVerlet))
        # results.append(solve_and_plot_earth_jupiter_and_sun_orbit(problem, VelocityVerlet, T=100))
        results.append(solve_and_plot_all_planet_orbits(problem, VelocityVerlet, T=100, names = PLANETS.keys()))
    table = pd.DataFrame(results)
    table.to_csv('test_stability_problem.csv')
    plt.show('hold')


if __name__ == '__main__':
    test_stability()
    # initial_conditions_solar_system()
