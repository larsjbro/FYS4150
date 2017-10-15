'''
Created on 2. okt. 2017

@author: LJB
'''
from __future__ import absolute_import, division
import numpy as np
import matplotlib.pyplot as plt
import unittest
from FYS4150.Project_2.source.project_2 import (jacobi_method)
from networkx.generators.threshold import eigenvalues


class TestJacobiOnSchrodinger(unittest.TestCase):

    def setUp(self):
        n = 128*4
        n = 320
        rho_min = 0
        rho_max = 5
        h = (rho_max - rho_min) / (n + 1)  # step_length
        rho = np.arange(1, n + 1) * h
        vi = rho**2

        self.rho = rho
        self.e = -np.ones(n - 1) / h**2
        self.d = 2 / h**2 + vi   #di
        self.A = np.diag(self.d) + np.diag(self.e, -1) + np.diag(self.e, +1)

        # Solve Schrodingers equation:
        self.eigenvectors, self.eigenvalues, self.iterations = jacobi_method(self.A)
        # self.eigenvalues, self.eigenvectors = np.linalg.eig(self.A)


    def test_three_first_eigenvalues(self):
        print sorted(self.eigenvalues)
        lamba3 = sorted(self.eigenvalues)[:3]
        np.testing.assert_allclose(lamba3, [3, 7, 11], rtol=1e-4)

    def test_jacobi_eigenvectors_are_orthogonal(self):
        U = self.eigenvectors
        # eigenvalues, eigenvectors = np.linalg.eig(self.A)
        n = len(U)
        for i in range(n):
            for j in range(i, n):
                true_val = 1 if i == j else 0
                np.testing.assert_allclose(np.dot(U[:, i], U[:, j]), true_val, atol=1e-7)

    def test_jacobi_solves_eigen_value_eq(self):
        """
        assert  A*u_j = lambda * u_j
        """
        A = self.A
        U = self.eigenvectors
        lambda_ = self.eigenvalues
        for k in range(len(A)):
            left = np.dot(A, U[:, k])
            right = lambda_[k] * U[:, k]
            # print(np.abs(left-right).max())
            np.testing.assert_allclose(left, right, atol=1e-4, rtol=1e-4)
            
    
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
