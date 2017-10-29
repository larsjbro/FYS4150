'''
Created on 25. okt. 2017

@author: LJB
'''
from __future__ import absolute_import, division
from ..ode_solvers import registered_solver_classes
import numpy as np
import unittest


class Test(unittest.TestCase):


    #def setUp(self):
        #pass


    #def tearDown(self):
        #pass


    def test_all_ode_solvers(self):
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
            classname = solver.__class__.__name__
    
            order = dict_order.get(classname, 1)
    
            t0 = timer()
            u, t = solver.solve(t_points)
            print('{} method used {} seconds'.format(classname, timer() - t0))
            u_e = u_exact(t)
            max_error = (u_e - u).max()
            msg = '%s failed with max_error=%g' % (classname, max_error)
            self.assertTrue(max_error < tol, msg)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()