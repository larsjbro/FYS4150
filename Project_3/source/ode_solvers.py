from numba import jit, autojit, jitclass
import numpy as np
# import sys
import os


class ODESolver(object):
    """
    Superclass for numerical methods solving scalar and vector ODEs

      du/dt = f(u, t)

    Attributes:
    t: array of time values
    u: array of solution values (at time points t)
    k: step number of the most recently computed solution
    f: callable object implementing f(u, t)
    """

    def __init__(self, f):

        # For ODE systems, f will often return a list, but
        # arithmetic operations with f in numerical methods
        # require that f is an array. Let self.f be a function
        # that first calls f(u,t) and then ensures that the
        # result is an array of floats.
        self.f = lambda u, t: np.asarray(f(u, t), float)

    @property
    def name(self):
        return self.__class__.__name__

    def advance(self, u, t, k):
        """Advance solution one time step from time step k to k+1."""
        raise NotImplementedError

    def _check_f(self):
        """Check that f returns correct length:"""
        f = self.f
        if not callable(f):
            raise TypeError('f is %s, not a function' % type(f))
        try:
            f0 = f(self.U0, 0)
        except IndexError:
            raise IndexError('Index of u out of bounds in f(u,t) func. '
                             'Legal indices are %s' % (str(range(self.neq))))
        if np.size(f0) != self.neq:
            raise ValueError('f(u,t) returns %d components, while u '
                             'has %d components' % (f0.size, self.neq))

    def set_initial_condition(self, U0):
        if isinstance(U0, (float, int)):  # scalar ODE
            self.neq = 1
            U0 = float(U0)
        else:                            # system of ODEs
            U0 = np.asarray(U0)          # (assume U0 is sequence)
            self.neq = U0.size
        self.U0 = U0

        self._check_f()

    def _check_time(self, time_points):
        if isinstance(time_points, (float, int)):
            raise TypeError('solve: time_points is not a sequence')
        if time_points.size <= 1:
            raise ValueError('ODESolver.solve requires time_points array with '
                             'at least 2 time points')

    def _initialize_u(self, n):
        if self.neq == 1:  # scalar ODEs
            u = np.zeros(n)
        else:
            u = np.zeros((n, self.neq))  # systems of ODEs
        # Assume that t[0] corresponds to self.U0
        u[0] = self.U0
        return u

    def solve(self, time_points, terminate=None):
        """
        Compute solution u for t values in the list/array
        time_points, as long as terminate(u,t,step_no) is False.
        terminate(u,t,step_no) is a user-given function
        returning True or False. By default, a terminate
        function which always returns False is used.
        """
        if terminate is None:
            # pylint: disable=function-redefined, unused-argument
            def terminate(u, t, step_no):  # @UnusedVariable
                return False

        self._check_time(time_points)

        t = np.asarray(time_points)
        n = t.size
        u = self._initialize_u(n)
        return self._solve(u, t, n, terminate)

    def _solve(self, u, t, n, terminate):
        # Time loop
        for k in range(n - 1):
            u[k + 1] = self.advance(u, t, k)
            if terminate(u, t, k + 1):
                return u[:k + 2], t[:k + 2]
        return u, t


class ForwardEuler(ODESolver):

    def advance(self, u, t, k):
        f = self.f
        dt = t[k + 1] - t[k]
        u_new = u[k] + dt * f(u[k], t[k])
        return u_new


class CentralEuler(ODESolver):

    def advance(self, u, t, k):
        f = self.f
        dt = t[k + 1] - t[k]
        if k == 0:
            u_new = u[k] + dt * f(u[k], t[k])
        else:
            u_new = u[k - 1] + 2 * dt * f(u[k], t[k])
        return u_new


class Verlet(ODESolver):
    """
    Verlet integration (without velocities)
    class for numerical methods solving scalar and vector ODEs

      d^2u/dt^2 = f(u, t)

    """

    def set_initial_condition(self, U0, v0=0):
        """
        Parameters
        ----------
        U0: initial accellerations
        v0: initial velocities
        """
        self.v0 = v0
        super(Verlet, self).set_initial_condition(U0)

    def advance(self, u, t, k):
        f = self.f
        dt = t[k + 1] - t[k]

        if k == 0:
            v0 = self.v0
            u_new = u[k] + dt * (v0 + 0.5 * dt * f(u[k], t[k]))
        else:
            u_new = 2 * u[k] - u[k - 1] + dt**2 * f(u[k], t[k])
        return u_new


class VelocityVerlet(Verlet):
    """
    class for numerical methods solving scalar and vector ODEs

      d^2u/dt^2 = f(u, t)

    """

    def advance(self, u, t, k):
        f = self.f
        dt = t[k + 1] - t[k]
        if k == 0:
            self.v = np.zeros_like(u)
            self.v[k] = self.v0

        a_k = f(u[k], t[k])
        v_k = self.v[k]
        u_new = u[k] + dt * (v_k + 0.5 * dt * a_k)
        a_kp1 = f(u_new, t[k + 1])
        self.v[k + 1] = v_k + 0.5 * dt * (a_k + a_kp1)
        return u_new


class RungeKutta4(ODESolver):

    def advance(self, u, t, k):
        f = self.f
        dt = t[k + 1] - t[k]
        dt2 = dt / 2.0
        K1 = dt * f(u[k], t[k])
        K2 = dt * f(u[k] + 0.5 * K1, t[k] + dt2)
        K3 = dt * f(u[k] + 0.5 * K2, t[k] + dt2)
        K4 = dt * f(u[k] + K3, t[k] + dt)
        u_new = u[k] + (1 / 6.0) * (K1 + 2 * K2 + 2 * K3 + K4)
        return u_new


class BackwardEuler(ODESolver):
    """Backward Euler solver for scalar ODEs."""

    def __init__(self, f):
        ODESolver.__init__(self, f)
        # Make a sample call to check that f is a scalar function:
        try:
            u = np.array([1])
            t = 1
            _value = f(u, t)
        except IndexError:  # index out of bounds for u
            raise ValueError('f(u,t) must return float/int')

        # BackwardEuler needs to import function Newton from Newton.py:
        try:
            from newton import Newton
            self.Newton = Newton
        except ImportError:
            raise ImportError('''
Could not import module "Newton". Place Newton.py in this directory
(%s)
''' % (os.path.dirname(os.path.abspath(__file__))))

    # Alternative implementation of F:
    # def F(self, w):
    #     return w - self.dt*self.f(w, self.t[-1]) - self.u[self.k]

    def advance(self, u, t, k):
        f = self.f
        dt = t[k + 1] - t[k]

        def F(w):
            return w - dt * f(w, t[k + 1]) - u[k]

        dFdw = Derivative(F)
        w_start = u[k] + dt * f(u[k], t[k])  # Forward Euler step
        u_new, n, _F_value = self.Newton(F, w_start, dFdw, N=30)
        if k == 0:
            self.Newton_iter = []
        self.Newton_iter.append(n)
        if n >= 30:
            print "Newton's failed to converge at t=%g "\
                  "(%d iterations)" % (t[k + 1], n)
        return u_new


class Derivative:

    def __init__(self, f, h=1E-9):
        self.f = f
        self.h = float(h)

    def __call__(self, x):
        f, h = self.f, self.h      # make short forms
        return (f(x + h) - f(x - h)) / (2 * h)


registered_solver_classes = [ForwardEuler, RungeKutta4, BackwardEuler, CentralEuler,
                             Verlet, VelocityVerlet]


def test_exact_numerical_solution():
    from timeit import default_timer as timer
    a = 0.2
    b = 3
    order = 1

    def f(u, t):
        return a + (u - u_exact(t))**5

    def u_exact(t):
        """Exact u(t) corresponding to f above."""
        return a / order * t**order + b

    U0 = u_exact(0)
    T = 8
    n = 10
    tol = 1E-14
    t_points = np.linspace(0, T, n)

    dict_order = dict(Verlet=2, VelocityVerlet=2)
    for solver_class in registered_solver_classes:

        solver = solver_class(f)
        solver.set_initial_condition(U0)
        classname = solver.name

        order = dict_order.get(classname, 1)

        t0 = timer()
        u, t = solver.solve(t_points)
        print('{} method used {} seconds'.format(classname, timer() - t0))
        u_e = u_exact(t)
        max_error = (u_e - u).max()
        msg = '%s failed with max_error=%g' % (classname, max_error)
        assert max_error < tol, msg


if __name__ == '__main__':
    test_exact_numerical_solution()
