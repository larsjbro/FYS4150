'''
Created on 26. okt. 2017

@author: LJB
'''
import numpy as np


class EarthSunSystem(object):
    def __init__(self, U0, dt, gm=4 * np.pi**2, beta=2):
        self.U0 = U0
        self.dt = dt
        self.gm = gm
        self.beta = beta

    def __call__(self, U, t):
        """Returns the right hand side of the differential equation

        dU/dt = f(U, t)

        where U = [x, y, dx/dt, dy/dt]

        Returns  dU/dt = [dx/dt, dy/dt, d^2x/dt^2, d^2y/dt^2]

        """
        x, y, dx, dy = U
        r3 = np.sqrt(x**2 + y**2)**(self.beta + 1)
        d2x = -self.gm * x / r3
        d2y = -self.gm * y / r3
        return dx, dy, d2x, d2y


class EarthSunSystem2(object):
    def __init__(self, U0, dt, gm=4 * np.pi**2, beta=2, v0=1):
        self.U0 = U0
        self.dt = dt
        self.gm = gm
        self.beta = beta
        self.v0 = v0

    def __call__(self, U, t):
        """Returns the right hand side of the differential equation

        d^2U/dt^2 = f(U, t)

        where U = [x, y]

        Returns  d**2U/dt**2 = [d^2x/dt^2, d^2y/dt^2]

        """
        x, y = U
        r3 = np.sqrt(x**2 + y**2)**(self.beta + 1)
        d2x = -self.gm * x / r3
        d2y = -self.gm * y / r3
        return d2x, d2y


class EarthSunJupiterSystem(object):
    """
    Parameters
    ----------
    U0: startpositions
    dt: sampling step
    gm: [Earth_mass, Jupiter_mass]/Sun_mass
    """

    def __init__(self, U0, dt, gm=(3.0e-6, 0.95e-3), beta=2, v0=1):
        self.U0 = U0
        self.dt = dt
        self.gm = gm
        self.beta = beta
        self.v0 = v0

    def __call__(self, U, t):
        """Returns the right hand side of the differential equation

        d^2U/dt^2 = f(U, t)

        where U = [x, y, z, xj, yj, zj]

        x, y, z is the position of Earth
        xj, yj, zj is the postiion of Jupiter

        Returns  d**2U/dt**2 = [d^2x/dt^2, d^2y/dt^2, d^2z/dt^2,
                                d^2xj/dt^2, d^2yj/dt^2 d^2zj/dt^2,]


        """
        x, y, z, xj, yj, zj = U

        r3 = np.sqrt(x**2 + y**2 + z**2)**(self.beta + 1)
        rj3 = np.sqrt(xj**2 + yj**2 + zj**2)**(self.beta + 1)
        rej3 = np.sqrt((x - xj)**2 + (x - yj)**2 + (z - zj)**2)**(self.beta + 1)

        earth_mass, jupiter_mass = self.gm  # relative to the sun mass
        four_pi2 = 4 * np.pi ** 2
        d2x = -four_pi2 * (x / r3 + jupiter_mass * (x - xj) / rej3)
        d2y = -four_pi2 * (y / r3 + jupiter_mass * (y - yj) / rej3)
        d2z = -four_pi2 * (z / r3 + jupiter_mass * (z - zj) / rej3)
        d2xj = -four_pi2 * (xj / rj3 + earth_mass * (xj - x) / rej3)
        d2yj = -four_pi2 * (yj / rj3 + earth_mass * (yj - y) / rej3)
        d2zj = -four_pi2 * (zj / rj3 + earth_mass * (zj - z) / rej3)
        return d2x, d2y, d2z, d2xj, d2yj, d2zj


class SolarSystem(object):
    """
    Solves the general model for all planets of the solar system.

    Parameters
    ----------
    U0: array_like. size 3*N
        Startpositions for the planets
    dt: real scalar
        sampling step
    gm: array_like, size N
        Relative planet masses [Sun mass,  Earth_mass, Jupiter_mass, ....]/Sun_mass
    v0: array_like, size 3*N
        Start velocities for the planets
    """

    def __init__(self, U0, dt, gm=(3.0e-6, 0.95e-3, 1), beta=2, v0=1):
        self.U0 = U0
        self.dt = dt
        self.gm = gm
        self.beta = beta
        self.v0 = v0

    def __call__(self, U, t):
        """Returns the right hand side of the differential equation

        d^2U/dt^2 = f(U, t)

        where U = [x1, y1, z1, x2, y2, z3, .... xn, yn, zn]
        and  xi, yi, zi is the position of planet number i

        Returns
        -------
        d**2U/dt**2 = [d^2x1/dt^2, d^2y1/dt^2, d^2z1/dt^2,
                       d^2x2/dt^2, d^2y2/dt^2 d^2z2/dt^2,
                       .....
                       ....
                       d^2xn/dt^2, d^2yn/dt^2 d^2zn/dt^2,]


        """
        masses = self.gm
        num_planets = len(masses)
        x = np.asarray(U).reshape((num_planets, 3))
        norm = np.linalg.norm
        dx = np.zeros((num_planets, 3))
        for i in range(num_planets):
            for j in range(i+1,num_planets):
                #if i!=j:
                rij = (x[i] - x[j])
                rij3 = norm(rij)**3
                dx[i] -= masses[j] * rij / rij3
                dx[j] += masses[i] * rij / rij3

        dx *= 4 * np.pi ** 2
        return dx.ravel()


if __name__ == '__main__':
    pass
