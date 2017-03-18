# -*- coding: utf-8 -*-
#The MIT License (MIT)
#
#Copyright (c) 2016 Peter Foster <pyitlib@gmx.us>
#
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
from scipy.stats.distributions import norm
import unittest
from pyitlib import discrete_random_variable as discrete

#TODO Is there a neater, more methodological way to group tests within each method?
#TODO Re-arrange functions

class TestEntropy(unittest.TestCase):
    def test_entropy_pmf(self):
        #Immutability test
        X1 = np.random.randint(16, size=(10,10))
        X1 = X1 / (1.0 * np.sum(X1, axis=1)[:,np.newaxis])
        X2 = np.copy(X1)
        discrete.entropy_pmf(X2)
        self.assertTrue(np.all(X2 == X1))
        
        #Basic tests
        self.assertTrue(discrete.entropy_pmf(np.array((0.5, 0.5)), base=2) == 1)
        self.assertTrue(discrete.entropy_pmf(np.array((1.0, 0)), base=2) == 0)
        self.assertTrue(abs(discrete.entropy_pmf(np.array((0.5, 0.5)), base=np.exp(1))-0.693) < 1E-03)
        
        #Type tests        
        self.assertTrue(isinstance(discrete.entropy_pmf(np.array((0.5, 0.5))) , np.float))
        self.assertTrue(isinstance(discrete.entropy_pmf(1) , np.float))
        
        #Output dimensionality tests -- vectors
        self.assertTrue(discrete.entropy_pmf(np.array(((1.0,), (1.0,), (1.0,)))).shape == (3,))
        self.assertTrue(discrete.entropy_pmf(np.array(((1.0/3,), (1.0/3,), (1.0/3,))).T).shape == (1,))
        
        self.assertTrue(discrete.entropy_pmf(np.array(1)) == 0)
        self.assertTrue(discrete.entropy_pmf(np.array((1,))) == 0)
        self.assertTrue(discrete.entropy_pmf(np.ones((1,))) == 0)
        self.assertTrue(discrete.entropy_pmf(np.ones((1,1))) == 0)
        
        X = np.ones((3,4,5))
        X = X / (1.0 * np.sum(X, axis=-1)[:,:,np.newaxis])
        self.assertTrue(isinstance(discrete.entropy_pmf(X), np.ndarray))
        self.assertTrue(discrete.entropy_pmf(X).shape == (3,4))                
        
        P1 = 1.0 * np.ones(1E06)
        P2 = 1.0 * np.ones(2E06)
        H1 = discrete.entropy_pmf(P1 / np.sum(P1), base=2)
        H2 = discrete.entropy_pmf(P2 / np.sum(P2), base=2)
        self.assertTrue(np.allclose(H1+1, H2))
                
        np.random.seed(4759)
        X1 = np.random.randn(10**6)
        X2 = X1 * 0.5;
        Bins1 = np.linspace(-4,4,10000)
        P1 = 1.0 * np.bincount(np.digitize(X1, Bins1))
        P2 = 1.0 * np.bincount(np.digitize(X2, Bins1))
        H1 = discrete.entropy_pmf(P1 / np.sum(P1), base=2)
        H2 = discrete.entropy_pmf(P2 / np.sum(P2), base=2)
        self.assertTrue(abs(H1 - 1 - H2) < 1E-02)
        
        #Exception tests
        with self.assertRaises(ValueError):
            discrete.entropy_pmf(np.array(()))
        with self.assertRaises(ValueError):
            discrete.entropy_pmf(np.array((np.nan, 1)))
        with self.assertRaises(ValueError):
            discrete.entropy_pmf(np.array((-1, 1)))
        with self.assertRaises(ValueError):
            discrete.entropy_pmf(np.array((1.1, 1.0)))
        with self.assertRaises(ValueError):
            discrete.entropy_pmf(np.array((0.5, 0.6))) 
        with self.assertRaises(ValueError):
            discrete.entropy_pmf(np.array((0.5, 0.5)), base=-1)                    
            
    def test_entropy_cross_pmf(self):
        #Immutability test
        X1 = np.random.randint(16, size=(10,10))
        X1 = X1 / (1.0 * np.sum(X1, axis=1)[:,np.newaxis])
        X1_copy = np.copy(X1)
        X2 = np.random.randint(16, size=(10,10))
        X2 = X2 / (1.0 * np.sum(X2, axis=1)[:,np.newaxis])
        X2_copy = np.copy(X2)        
        discrete.entropy_cross_pmf(X1_copy, X2_copy)
        self.assertTrue(np.all(X1_copy == X1) and np.all(X2_copy == X2))
               
        #Basic tests
        self.assertTrue(discrete.entropy_cross_pmf(np.array((1.0, 0)), np.array((0.5, 0.5)), base=2) == 1)
        self.assertTrue(discrete.entropy_cross_pmf(np.array((1.0, 0)), np.array((1.0, 0)), base=2) == 0)        
        self.assertTrue(discrete.entropy_cross_pmf(np.array((0.5, 0.5)), np.array((0.5, 0.5)), base=2) == 1)
        self.assertTrue(abs(discrete.entropy_cross_pmf(np.array((0.5, 0.5)), np.array((0.5, 0.5)), base=np.exp(1))-0.693) < 1E-03)
        self.assertTrue(abs(discrete.entropy_cross_pmf(np.array((1.0, 0.0)), np.array((0.5, 0.5)), base=np.exp(1))-0.693) < 1E-03)
        #Tests using cartesian_product=True
        self.assertTrue(discrete.entropy_cross_pmf(np.array((1.0, 0)), np.array((0.5, 0.5)), cartesian_product=True, base=2) == 1)
        self.assertTrue(discrete.entropy_cross_pmf(np.array((1.0, 0)), np.array((1.0, 0)), cartesian_product=True, base=2) == 0)        
        self.assertTrue(discrete.entropy_cross_pmf(np.array((0.5, 0.5)), np.array((0.5, 0.5)), cartesian_product=True, base=2) == 1)
        self.assertTrue(abs(discrete.entropy_cross_pmf(np.array((0.5, 0.5)), np.array((0.5, 0.5)), cartesian_product=True, base=np.exp(1))-0.693) < 1E-03)
        self.assertTrue(abs(discrete.entropy_cross_pmf(np.array((1.0, 0.0)), np.array((0.5, 0.5)), cartesian_product=True, base=np.exp(1))-0.693) < 1E-03)                
        
        #Type tests        
        self.assertTrue(isinstance(discrete.entropy_cross_pmf(np.array((0.5, 0.5)), np.array((0.5, 0.5))) , np.float))
        self.assertTrue(isinstance(discrete.entropy_cross_pmf(1,1) , np.float))
        
        #Output dimensionality tests -- vectors
        self.assertTrue(discrete.entropy_cross_pmf(np.array(((1.0,), (1.0,), (1.0,))), np.array(((1.0,), (1.0,), (1.0,)))).shape == (3,))
        self.assertTrue(discrete.entropy_cross_pmf(np.array(((1.0/3,), (1.0/3,), (1.0/3,))).T, np.array(((1.0/3,), (1.0/3,), (1.0/3,))).T).shape == (1,))
        #Tests using cartesian_product=True
        self.assertTrue(discrete.entropy_cross_pmf(np.array(((1.0,), (1.0,), (1.0,))), np.array(((1.0,), (1.0,), (1.0,))), cartesian_product=True).shape == (3,3))
        self.assertTrue(discrete.entropy_cross_pmf(np.array(((1.0/3,), (1.0/3,), (1.0/3,))).T, np.array(((1.0/3,), (1.0/3,), (1.0/3,))).T, cartesian_product=True).shape == (1,1))        

        self.assertTrue(discrete.entropy_cross_pmf(np.array(1), np.array(1)) == 0)
        self.assertTrue(discrete.entropy_cross_pmf(np.array((1,)), np.array((1,))) == 0)
        self.assertTrue(discrete.entropy_cross_pmf(np.ones((1,)), np.ones((1,))) == 0)
        self.assertTrue(discrete.entropy_cross_pmf(np.ones((1,1)), np.ones((1,1))) == 0)
        #Tests using cartesian_product=True
        self.assertTrue(discrete.entropy_cross_pmf(np.array(1), np.array(1), cartesian_product=True) == 0)
        self.assertTrue(discrete.entropy_cross_pmf(np.array((1,)), np.array((1,)), cartesian_product=True) == 0)
        self.assertTrue(discrete.entropy_cross_pmf(np.ones((1,)), np.ones((1,)),  cartesian_product=True) == 0)
        self.assertTrue(discrete.entropy_cross_pmf(np.ones((1,1)), np.ones((1,1)), cartesian_product=True) == 0)
                    
        self.assertTrue(np.all(discrete.entropy_cross_pmf(np.array(((0.5, 0.5), (0.5, 0.5))))==1))
        self.assertTrue(np.all(discrete.entropy_cross_pmf(np.array(((1.0, 0.0), (0.0, 1.0))),np.array(((1.0, 0.0), (0.0, 1.0))))==0))
        #Tests using cartesian_product=True
        self.assertTrue(np.all(discrete.entropy_cross_pmf(np.array(((0.5, 0.5), (0.5, 0.5))), cartesian_product=True)==1))
        self.assertTrue(np.all(discrete.entropy_cross_pmf(np.array(((1.0, 0.0), (0.0, 1.0))),np.array(((1.0, 0.0), (0.0, 1.0))), cartesian_product=True)==np.array(((0.0, np.inf),(np.inf, 0.0)))))
        
        self.assertTrue(discrete.entropy_cross_pmf(np.array(1),np.array(1)).shape == tuple())
        self.assertTrue(discrete.entropy_cross_pmf(np.array((1,)),np.array((1,))).shape == tuple())
        self.assertTrue(discrete.entropy_cross_pmf(np.ones((1,)),np.ones((1,))).shape == tuple())
        self.assertTrue(discrete.entropy_cross_pmf(np.ones((1,1)),np.ones((1,1))).shape == (1,))
        #Tests using cartesian_product=True
        self.assertTrue(discrete.entropy_cross_pmf(np.array(1),np.array(1),True).shape == tuple())
        self.assertTrue(discrete.entropy_cross_pmf(np.array((1,)),np.array((1,)),True).shape == tuple())
        self.assertTrue(discrete.entropy_cross_pmf(np.ones((1,)),np.ones((1,)),True).shape == tuple())
        self.assertTrue(discrete.entropy_cross_pmf(np.ones((1,1)),np.ones((1,1)),True).shape == (1,1))
        self.assertTrue(discrete.entropy_cross_pmf(np.ones((1,)),np.ones((1,1)),True).shape == (1,))
        self.assertTrue(discrete.entropy_cross_pmf(np.ones((1,1)),np.ones((1,)),True).shape == (1,))
        self.assertTrue(discrete.entropy_cross_pmf(np.ones((1,1)),np.ones((1,1)),True).shape == (1,1))
        
        X = np.ones((3,4,5))
        X = X / (1.0 * np.sum(X, axis=-1)[:,:,np.newaxis])
        self.assertTrue(isinstance(discrete.entropy_cross_pmf(X,X), np.ndarray))
        self.assertTrue(discrete.entropy_cross_pmf(X,X).shape == (3,4))
        #Tests using cartesian_product=True
        X = np.ones((3,4,5))
        X = X / (1.0 * np.sum(X, axis=-1)[:,:,np.newaxis])
        self.assertTrue(isinstance(discrete.entropy_cross_pmf(X,X,cartesian_product=True), np.ndarray))
        self.assertTrue(discrete.entropy_cross_pmf(X,X,cartesian_product=True).shape == (3,4,3,4))        
        
        #NB: Distribution tests implemented for divergence_kullbackleibler_pmf()        
        
        #Exception tests
        with self.assertRaises(ValueError):
            discrete.entropy_cross_pmf(np.array(()), np.array(()))
        with self.assertRaises(ValueError):
            discrete.entropy_cross_pmf(np.array((0.5,0.5)), np.array((1.0)))
        with self.assertRaises(ValueError):
            discrete.entropy_cross_pmf(np.array((1.1,-0.1)), np.array((1.0,0.0)))
        with self.assertRaises(ValueError):
            discrete.entropy_cross_pmf(np.array((0.5,0.5)), np.array((1.1,-0.1)))            
        with self.assertRaises(ValueError):
            discrete.entropy_cross_pmf(np.array((np.nan, 1)), np.array((0.5,0.5)))
        with self.assertRaises(ValueError):
            discrete.entropy_cross_pmf(np.array((0.5, 0.5)), np.array((np.nan, 1.0)))
        with self.assertRaises(ValueError):
            discrete.entropy_cross_pmf(np.array((0.5, 0.5)), np.array((0.5, 0.5, 0.0)))
        with self.assertRaises(ValueError):
            discrete.entropy_cross_pmf(np.array((0.5, 0.5)), np.array((0.5, 0.5, 0.0)), True)            
        try:
            discrete.entropy_cross_pmf(np.array(((0.5,0.5),(0.5, 0.5))), np.array((0.5, 0.5)), True)            
        except ValueError:
            self.fail("raised ValueError unexpectedly")
        with self.assertRaises(ValueError):
            discrete.entropy_cross_pmf(np.array((0.5,0.6)), np.array((0.5,0.5)))            
        with self.assertRaises(ValueError):
            discrete.entropy_cross_pmf(np.array((0.5,0.5)), np.array((0.5,0.6)))
        with self.assertRaises(ValueError):
            discrete.entropy_cross_pmf(np.array((0.5, 0.5)), np.array((0.5, 0.5)), base=-1)
            
    def test_divergence_kullbackleibler_pmf(self):
        self.assertTrue(discrete.divergence_kullbackleibler_pmf(np.array(1),np.array(1)).shape == tuple())
        self.assertTrue(discrete.divergence_kullbackleibler_pmf(np.array((1,)),np.array((1,))).shape == tuple())
        self.assertTrue(discrete.divergence_kullbackleibler_pmf(np.ones((1,)),np.ones((1,))).shape == tuple())
        self.assertTrue(discrete.divergence_kullbackleibler_pmf(np.ones((1,1)),np.ones((1,1))).shape == (1,))
        #Tests using cartesian_product=True
        self.assertTrue(discrete.divergence_kullbackleibler_pmf(np.array(1),np.array(1),True).shape == tuple())
        self.assertTrue(discrete.divergence_kullbackleibler_pmf(np.array((1,)),np.array((1,)),True).shape == tuple())
        self.assertTrue(discrete.divergence_kullbackleibler_pmf(np.ones((1,)),np.ones((1,)),True).shape == tuple())
        self.assertTrue(discrete.divergence_kullbackleibler_pmf(np.ones((1,1)),np.ones((1,1)),True).shape == (1,1))
        self.assertTrue(discrete.divergence_kullbackleibler_pmf(np.ones((1,)),np.ones((1,1)),True).shape == (1,))
        self.assertTrue(discrete.divergence_kullbackleibler_pmf(np.ones((1,1)),np.ones((1,)),True).shape == (1,))
        self.assertTrue(discrete.divergence_kullbackleibler_pmf(np.ones((1,1)),np.ones((1,1)),True).shape == (1,1))
        
        P1 = 1.0 * np.ones(1E06)
        H1 = discrete.divergence_kullbackleibler_pmf(P1 / np.sum(P1), P1 / np.sum(P1), base=2)
        self.assertTrue(H1 == 0)

        Bins1 = np.linspace(-5,5,10000)
        P1 = norm.pdf(Bins1)
        P2 = norm.pdf(Bins1,loc=0.5)
        H1 = discrete.divergence_kullbackleibler_pmf(P1 / np.sum(P1), P2 / np.sum(P2), base=np.exp(1))
        self.assertTrue(np.abs(H1 - 0.125) < 1E-3)
        
        Locs = np.linspace(-0.5,0.5,64)
        P1 = [norm.pdf(Bins1) for i in np.arange(64)]
        P2 = [norm.pdf(Bins1,Locs[i]) for i in np.arange(64)]
        H1 = [(1 + loc**2)/2 - 0.5 for loc in Locs]              
        P1 = np.array(P1).reshape((8,8,-1))
        P2 = np.array(P2).reshape((8,8,-1))
        H1 = np.array(H1).reshape((8,8))
        H1_empirical = discrete.divergence_kullbackleibler_pmf(P1 / np.sum(P1,axis=-1)[:,:,np.newaxis], P2 / np.sum(P2,axis=-1)[:,:,np.newaxis], base=np.exp(1))
        self.assertTrue(np.all(H1_empirical.shape == (8,8)))
        self.assertTrue(np.all(np.abs(H1 - H1_empirical) < 1E-3))
        
        Locs = np.linspace(-0.5,0.5,64)
        P1 = [norm.pdf(Bins1,Locs[i]) for i in np.arange(64)]
        P2 = [norm.pdf(Bins1,Locs[i]) for i in np.arange(64)]
        P1 = np.array(P1).reshape((8,8,-1))
        P2 = np.array(P2).reshape((8,8,-1))
        Locs = Locs.reshape((8,8,-1))
        H1_empirical = discrete.divergence_kullbackleibler_pmf(P1 / np.sum(P1,axis=-1)[:,:,np.newaxis], P2 / np.sum(P2,axis=-1)[:,:,np.newaxis], cartesian_product=True, base=np.exp(1))        
        self.assertTrue(np.all(H1_empirical.shape == (8,8,8,8)))
        for i in xrange(H1_empirical.shape[0]):
            for j in xrange(H1_empirical.shape[1]):
                for k in xrange(H1_empirical.shape[2]):
                   for l in xrange(H1_empirical.shape[3]):
                       H1 = (1 + (Locs[i,j]-Locs[k,l])**2)/2 - 0.5
                       self.assertTrue(np.abs(H1 - H1_empirical[i,j,k,l]) < 1E-3)
                       
    def test_divergence_kullbackleibler_symmetrised_pmf(self):
        self.assertTrue(discrete.divergence_kullbackleibler_symmetrised_pmf(np.array(1),np.array(1)).shape == tuple())
        self.assertTrue(discrete.divergence_kullbackleibler_symmetrised_pmf(np.array((1,)),np.array((1,))).shape == tuple())
        self.assertTrue(discrete.divergence_kullbackleibler_symmetrised_pmf(np.ones((1,)),np.ones((1,))).shape == tuple())
        self.assertTrue(discrete.divergence_kullbackleibler_symmetrised_pmf(np.ones((1,1)),np.ones((1,1))).shape == (1,))
        #Tests using cartesian_product=True
        self.assertTrue(discrete.divergence_kullbackleibler_symmetrised_pmf(np.array(1),np.array(1),True).shape == tuple())
        self.assertTrue(discrete.divergence_kullbackleibler_symmetrised_pmf(np.array((1,)),np.array((1,)),True).shape == tuple())
        self.assertTrue(discrete.divergence_kullbackleibler_symmetrised_pmf(np.ones((1,)),np.ones((1,)),True).shape == tuple())
        self.assertTrue(discrete.divergence_kullbackleibler_symmetrised_pmf(np.ones((1,1)),np.ones((1,1)),True).shape == (1,1))
        self.assertTrue(discrete.divergence_kullbackleibler_symmetrised_pmf(np.ones((1,)),np.ones((1,1)),True).shape == (1,))
        self.assertTrue(discrete.divergence_kullbackleibler_symmetrised_pmf(np.ones((1,1)),np.ones((1,)),True).shape == (1,))
        self.assertTrue(discrete.divergence_kullbackleibler_symmetrised_pmf(np.ones((1,1)),np.ones((1,1)),True).shape == (1,1))        
        
        self.assertTrue(np.allclose(discrete.divergence_kullbackleibler_symmetrised_pmf(np.array((2.0/3, 1.0/3)), np.array((1.0/3, 2.0/3))), 2.0/3))
        
        P = np.array(((2.0/3, 1.0/3),(1.0/3, 2.0/3),(2.0/3, 1.0/3)))
        Q = np.array(((2.0/3, 1.0/3),(1.0/3, 2.0/3)))
        H = discrete.divergence_kullbackleibler_symmetrised_pmf(P,Q, True)
        self.assertTrue(np.allclose(H, np.array(((0.0,2.0/3),(2.0/3,0.0),(0.0,2.0/3)))))
        H = discrete.divergence_kullbackleibler_symmetrised_pmf(Q)
        self.assertTrue(np.allclose(H, np.array(((0.0,2.0/3),(2.0/3,0.0)))))
        
    def test_divergence_jensenshannon_pmf(self):
        #Immutability test
        X1 = np.random.randint(16, size=(10,10))
        X1 = X1 / (1.0 * np.sum(X1, axis=1)[:,np.newaxis])
        X1_copy = np.copy(X1)
        X2 = np.random.randint(16, size=(10,10))
        X2 = X2 / (1.0 * np.sum(X2, axis=1)[:,np.newaxis])
        X2_copy = np.copy(X2)        
        discrete.divergence_jensenshannon_pmf(X1_copy, X2_copy)
        self.assertTrue(np.all(X1_copy == X1) and np.all(X2_copy == X2))
               
        #Basic tests
        self.assertTrue(discrete.divergence_jensenshannon_pmf(np.array((1.0, 0)), np.array((0, 1.0)), base=2) == 1)
        self.assertTrue(discrete.divergence_jensenshannon_pmf(np.array((1.0, 0)), np.array((1.0, 0)), base=2) == 0)        
        self.assertTrue(discrete.divergence_jensenshannon_pmf(np.array((0.5, 0.5)), np.array((0.5, 0.5)), base=2) == 0)
        self.assertTrue(abs(discrete.divergence_jensenshannon_pmf(np.array((1.0, 0.0)), np.array((0.0, 1.0)), base=np.exp(1))-0.693) < 1E-03)
        #Tests using cartesian_product=True
        self.assertTrue(discrete.divergence_jensenshannon_pmf(np.array((1.0, 0)), np.array((0, 1.0)), cartesian_product=True, base=2) == 1)
        self.assertTrue(discrete.divergence_jensenshannon_pmf(np.array((1.0, 0)), np.array((1.0, 0)), cartesian_product=True, base=2) == 0)        
        self.assertTrue(discrete.divergence_jensenshannon_pmf(np.array((0.5, 0.5)), np.array((0.5, 0.5)), cartesian_product=True, base=2) == 0)
        self.assertTrue(abs(discrete.divergence_jensenshannon_pmf(np.array((1.0, 0.0)), np.array((0.0, 1.0)), cartesian_product=True, base=np.exp(1))-0.693) < 1E-03)
        
        #Type tests        
        self.assertTrue(isinstance(discrete.divergence_jensenshannon_pmf(np.array((0.5, 0.5)), np.array((0.5, 0.5))) , np.float))
        self.assertTrue(isinstance(discrete.divergence_jensenshannon_pmf(1,1) , np.float))
        
        #Output dimensionality tests -- vectors
        self.assertTrue(discrete.divergence_jensenshannon_pmf(np.array(((1.0,), (1.0,), (1.0,))), np.array(((1.0,), (1.0,), (1.0,)))).shape == (3,))
        self.assertTrue(discrete.divergence_jensenshannon_pmf(np.array(((1.0/3,), (1.0/3,), (1.0/3,))).T, np.array(((1.0/3,), (1.0/3,), (1.0/3,))).T).shape == (1,))
        #Tests using cartesian_product=True
        self.assertTrue(discrete.divergence_jensenshannon_pmf(np.array(((1.0,), (1.0,), (1.0,))), np.array(((1.0,), (1.0,), (1.0,))), cartesian_product=True).shape == (3,3))
        self.assertTrue(discrete.divergence_jensenshannon_pmf(np.array(((1.0/3,), (1.0/3,), (1.0/3,))).T, np.array(((1.0/3,), (1.0/3,), (1.0/3,))).T, cartesian_product=True).shape == (1,1))        

        self.assertTrue(discrete.divergence_jensenshannon_pmf(np.array(1), np.array(1)) == 0)
        self.assertTrue(discrete.divergence_jensenshannon_pmf(np.array((1,)), np.array((1,))) == 0)
        self.assertTrue(discrete.divergence_jensenshannon_pmf(np.ones((1,)), np.ones((1,))) == 0)
        self.assertTrue(discrete.divergence_jensenshannon_pmf(np.ones((1,1)), np.ones((1,1))) == 0)
        #Tests using cartesian_product=True
        self.assertTrue(discrete.divergence_jensenshannon_pmf(np.array(1), np.array(1), cartesian_product=True) == 0)
        self.assertTrue(discrete.divergence_jensenshannon_pmf(np.array((1,)), np.array((1,)), cartesian_product=True) == 0)
        self.assertTrue(discrete.divergence_jensenshannon_pmf(np.ones((1,)), np.ones((1,)),  cartesian_product=True) == 0)
        self.assertTrue(discrete.divergence_jensenshannon_pmf(np.ones((1,1)), np.ones((1,1)), cartesian_product=True) == 0)
                    
        self.assertTrue(np.all(discrete.divergence_jensenshannon_pmf(np.array(((0.5, 0.5), (0.5, 0.5))))==0))
        self.assertTrue(np.all(discrete.divergence_jensenshannon_pmf(np.array(((1.0, 0.0), (0.0, 1.0))),np.array(((1.0, 0.0), (0.0, 1.0))))==0))
        #Tests using cartesian_product=True
        self.assertTrue(np.all(discrete.divergence_jensenshannon_pmf(np.array(((1.0, 0.0), (0.0, 1.0))), cartesian_product=True)==np.array(((0.0, 1),(1, 0.0)))))
        self.assertTrue(np.all(discrete.divergence_jensenshannon_pmf(np.array(((1.0, 0.0), (0.0, 1.0))),np.array(((1.0, 0.0), (0.0, 1.0))), cartesian_product=True)==np.array(((0.0, 1),(1, 0.0)))))
        
        #Test based on interpreting JSD as mutual information between X, sourced from mixture distribution, and Y, the mixture component indicator 
        Bins1 = np.linspace(-5,5,10000)
        P1 = norm.pdf(Bins1,loc=-0.5)
        P2 = norm.pdf(Bins1,loc=1.0)
        P1 = P1 / np.sum(P1)
        P2 = P2 / np.sum(P2)
        P = np.append(P1, P2)
        P = P / np.sum(P)
        #Sample from component distributions
        I = np.random.choice(P.size, 10**6, p=P)
        #Get component indicator
        C = I >= 10000
        #Mod operation to complete mixture sampling
        I = I % 10000
        #Compute JSD based on component pmfs
        JSD = discrete.divergence_jensenshannon_pmf(P1, P2)
        #Estimate mutual information between C and I
        MI = discrete.entropy(C) - discrete.entropy_conditional(C, I)
        self.assertTrue(np.abs(JSD - MI) < 0.01)
        
        #Additional test based on Kullback-Leibler divergence
        M = 0.5 * (P1 + P2)
        JSD2 = 0.5 * (discrete.divergence_kullbackleibler_pmf(P1, M) + discrete.divergence_kullbackleibler_pmf(P2, M))
        self.assertTrue(np.allclose(JSD2, JSD))
        
        #Exception tests
        with self.assertRaises(ValueError):
            discrete.divergence_jensenshannon_pmf(np.array(()), np.array(()))
        with self.assertRaises(ValueError):
            discrete.divergence_jensenshannon_pmf(np.array((0.5,0.5)), np.array((1.0)))
        with self.assertRaises(ValueError):
            discrete.divergence_jensenshannon_pmf(np.array((1.1,-0.1)), np.array((1.0,0.0)))
        with self.assertRaises(ValueError):
            discrete.divergence_jensenshannon_pmf(np.array((0.5,0.5)), np.array((1.1,-0.1)))            
        with self.assertRaises(ValueError):
            discrete.divergence_jensenshannon_pmf(np.array((np.nan, 1)), np.array((0.5,0.5)))
        with self.assertRaises(ValueError):
            discrete.divergence_jensenshannon_pmf(np.array((0.5, 0.5)), np.array((np.nan, 1.0)))
        with self.assertRaises(ValueError):
            discrete.divergence_jensenshannon_pmf(np.array((0.5, 0.5)), np.array((0.5, 0.5, 0.0)))
        with self.assertRaises(ValueError):
            discrete.divergence_jensenshannon_pmf(np.array((0.5, 0.5)), np.array((0.5, 0.5, 0.0)), True)            
        try:
            discrete.divergence_jensenshannon_pmf(np.array(((0.5,0.5),(0.5, 0.5))), np.array((0.5, 0.5)), True)            
        except ValueError:
            self.fail("raised ValueError unexpectedly")
        with self.assertRaises(ValueError):
            discrete.divergence_jensenshannon_pmf(np.array((0.5,0.6)), np.array((0.5,0.5)))            
        with self.assertRaises(ValueError):
            discrete.divergence_jensenshannon_pmf(np.array((0.5,0.5)), np.array((0.5,0.6)))
        with self.assertRaises(ValueError):
            discrete.divergence_jensenshannon_pmf(np.array((0.5, 0.5)), np.array((0.5, 0.5)), base=-1)
                               
    def test_entropy(self):
        #Immutability test
        X1 = np.random.randint(16, size=(10,10))
        X2 = np.copy(X1)
        discrete.entropy(X2)
        self.assertTrue(np.all(X2 == X1))
        
        self.assertTrue(discrete.entropy(np.array((0.5, 0.5, 1.0, 1.0)), base=2) == 1)
        self.assertTrue(discrete.entropy(np.array((1.0, 1.0)), base=2) == 0)
        self.assertTrue(abs(discrete.entropy(np.array((1.0, 2.0)), base=np.exp(1))-0.693) < 1E-03)
        
        self.assertTrue(discrete.entropy(np.ones(5))==0)
        self.assertTrue(discrete.entropy(np.ones((5,1)).T)==0)
        self.assertTrue(discrete.entropy(np.ones((5,1))).size==5)
        self.assertTrue(np.all(discrete.entropy(np.ones((5,1)))==0))
        
        self.assertTrue(discrete.entropy(np.array(4)) == 0)
        self.assertTrue(discrete.entropy(np.array((4,))) == 0)
        self.assertTrue(discrete.entropy(np.ones((1,))) == 0)
        self.assertTrue(discrete.entropy(np.ones((1,1))) == 0)        
        
        self.assertTrue(np.all(discrete.entropy(np.array(((1.0, 2.0), (1.0, 2.0))))==1))
        self.assertTrue(np.all(discrete.entropy(np.array(((1.0, 2.0), (1.0, 2.0))).T)==0))
        
        self.assertTrue(discrete.entropy(np.array(1)).shape == tuple())
        self.assertTrue(discrete.entropy(np.array((1,))).shape == tuple())
        self.assertTrue(discrete.entropy(np.ones((1,))).shape == tuple())
        self.assertTrue(discrete.entropy(np.ones((1,1))).shape == (1,))        
        
        X = np.arange(3*4*5).reshape((3,4,5))
        self.assertTrue(isinstance(discrete.entropy(X), np.ndarray))
        self.assertTrue(discrete.entropy(X).shape == (3,4))
        self.assertTrue(np.allclose(discrete.entropy(X), -np.log2(1.0/5)))
        
        np.random.seed(4759)
        X = np.random.randint(16, size=(10,10**4))
        self.assertTrue(np.all(np.abs(discrete.entropy(X) - 4.0) < 1E-02))
                
        np.random.seed(4759)
        X1 = np.random.randn(10**6)
        X2 = X1 * 0.5;
        Bins1 = np.linspace(-4,4,10000)
        P1 = np.digitize(X1, Bins1)
        P2 = np.digitize(X2, Bins1)
        H1 = discrete.entropy(P1, base=2)
        H2 = discrete.entropy(P2, base=2)
        self.assertTrue(abs(H1 - 1 - H2) < 1E-02)
        
        #Exception tests
        with self.assertRaises(ValueError):
            discrete.entropy(np.array(()))
        with self.assertRaises(ValueError):
            discrete.entropy(np.array((np.nan, 1)))
        with self.assertRaises(ValueError):
            discrete.entropy(np.array((1,1)), Alphabet_X=(1,2,np.nan))
        with self.assertRaises(ValueError):
            discrete.entropy(np.array((1,1)), fill_value=np.nan) 
        with self.assertRaises(ValueError):
            discrete.entropy(np.array((2, 1)), base=-1)
            
        #
        # Tests using missing data
        #
        #Immutability test
        X1 = np.random.randint(16, size=(10,10)) * 1.0
        X1[0,0] = -1
        X2 = np.copy(X1)
        discrete.entropy(X2, fill_value=-1)
        self.assertTrue(np.all(X2 == X1))
        
        # Tests using missing data involving masked arrays
        X1 = np.random.randint(16, size=(10,10)) * 1.0
        X1[0,0] = -1
        Mask = np.zeros_like(X1)
        Mask[0,0] = 1
        self.assertTrue(np.all(discrete.entropy(X1, fill_value=-1) == discrete.entropy(np.ma.array(X1,mask=Mask))))

        #Tests involving masked arrays        
        self.assertTrue(discrete.entropy(np.ma.array((4,1,2), mask=(0,1,0))) == 1)
        self.assertTrue(discrete.entropy(np.ma.array((4,1), mask=(0,0))) == 1)        
        self.assertTrue(discrete.entropy(np.ma.array((4.1,1.0,2.0), mask=(0,1,0))) == 1)
        self.assertTrue(discrete.entropy(np.ma.array(('a','b','c'), mask=(0,1,0))) == 1)
        self.assertTrue(discrete.entropy(np.ma.array(('N','b','c'), mask=(0,1,0))) == 1)
        self.assertTrue(discrete.entropy(np.ma.array(('N/','bb','cc'), mask=(0,1,0))) == 1)
        self.assertTrue(discrete.entropy(np.ma.array(('N/A.','two','three'), mask=(0,1,0))) == 1)
        self.assertTrue(np.isnan(discrete.entropy(np.ma.array((4.1,1.0,2.0), mask=(1,1,1)))))

        #Tests involving standard arrays
        self.assertTrue(discrete.entropy(np.array((4,None,2)), fill_value=None) == 1)
        self.assertTrue(discrete.entropy(np.array((4,1)), fill_value=None) == 1)        
        self.assertTrue(discrete.entropy(np.array((4.1,None,2.0)), fill_value=None) == 1)
        self.assertTrue(discrete.entropy(np.array(('a','N/A','c')), fill_value='N/A') == 1)
        self.assertTrue(discrete.entropy(np.array(('N','N/A','c')), fill_value='N/A') == 1)
        self.assertTrue(discrete.entropy(np.array(('N/','N/A','cc')), fill_value='N/A') == 1)
        self.assertTrue(discrete.entropy(np.array(('N/A.','N/A','three')), fill_value='N/A') == 1)
        self.assertTrue(np.isnan(discrete.entropy(np.array((None,None,None)), fill_value=None)))
        
        # Tests using missing data and None values
        self.assertTrue(discrete.entropy((1,2,1,2,1,1,None,None), fill_value=-1) == discrete.entropy((1,2,1,2,1,1,3,3)))
        self.assertTrue(discrete.entropy((1,2,1,2,1,1,None,None), fill_value=None) == discrete.entropy((1,2,1,2,1,1)))
        self.assertTrue(discrete.entropy(np.ma.array((1,2,1,2,1,1,None,None),mask=(0,0,0,0,0,0,1,1))) == discrete.entropy((1,2,1,2,1,1)))
        self.assertTrue(discrete.entropy((1,2,1,2,1,1), Alphabet_X=(1,2,None), fill_value=-1) == discrete.entropy((1,2,1,2,1,1), Alphabet_X=(1,2,3)))
        self.assertTrue(discrete.entropy((1,2,1,2,1,1), Alphabet_X=(1,2,None), fill_value=None) == discrete.entropy((1,2,1,2,1,1), Alphabet_X=(1,2)))
        self.assertTrue(discrete.entropy((1,2,1,2,1,1), Alphabet_X=np.ma.array((1,2,None),mask=(0,0,1))) == discrete.entropy((1,2,1,2,1,1), Alphabet_X=(1,2)))        
        self.assertTrue(discrete.entropy((1,2,1,2,1,1,None,None), Alphabet_X=(1,2,None), fill_value=-1) == discrete.entropy((1,2,1,2,1,1,3,3), Alphabet_X=(1,2,3)))
        self.assertTrue(discrete.entropy((1,2,1,2,1,1,None,None), Alphabet_X=(1,2,None), fill_value=None) == discrete.entropy((1,2,1,2,1,1), Alphabet_X=(1,2)))
        self.assertTrue(discrete.entropy(np.ma.array((1,2,1,2,1,1,None,None),mask=(0,0,0,0,0,0,1,1)), Alphabet_X=np.ma.array((1,2,3),mask=(0,0,1))) == discrete.entropy((1,2,1,2,1,1), Alphabet_X=(1,2)))        
        
        ### Tests involving Alphabet and non-ML estimation
        ###
        ###
        #Alphabet immutability 
        X1 = np.random.randint(16, size=(10,10))
        X2 = np.copy(X1)
        discrete.entropy(X2, estimator=1, Alphabet_X=X2)
        self.assertTrue(np.all(X2 == X1))
        #No explicit alphabet
        self.assertTrue(discrete.entropy([1, 2], estimator=1, Alphabet_X=None) == 1)
        #Larger alphabet
        self.assertTrue(np.abs(discrete.entropy([1, 2], estimator=1, Alphabet_X=[1,2,3], base=3)-0.96023)< 1E-03)
        #Empty alphabet
        with self.assertRaises(ValueError):
            discrete.entropy([1, 2], estimator=1, Alphabet_X=np.array(()), base=3)
        #Smaller alphabet
        with self.assertRaises(ValueError):
            discrete.entropy([1, 2], estimator=1, Alphabet_X=[1,], base=3)
        #Alphabet with incorrect dimensions
        with self.assertRaises(ValueError):
            discrete.entropy(np.array((1,2)), estimator=1, Alphabet_X=np.array(((1,2),(1,2))), base=3)
                
    def test_entropy_joint(self):
        #Immutability test
        X1 = np.random.randint(16, size=(10,10))
        X2 = np.copy(X1)
        discrete.entropy_joint(X2)
        self.assertTrue(np.all(X2 == X1))
        
        self.assertTrue(discrete.entropy_joint(np.array((0.5, 0.5, 1.0, 1.0)), base=2) == 1)
        self.assertTrue(discrete.entropy_joint(np.array((1.0, 1.0)), base=2) == 0)
        self.assertTrue(abs(discrete.entropy_joint(np.array((1.0, 2.0)), base=np.exp(1))-0.693) < 1E-03)
        
        self.assertTrue(discrete.entropy_joint(np.ones(5))==0)
        self.assertTrue(discrete.entropy_joint(np.ones((5,1)).T)==0)
        self.assertTrue(discrete.entropy_joint(np.ones((5,1)))==0)
        self.assertTrue(discrete.entropy_joint(np.ones((5,1)))==0)
        
        self.assertTrue(discrete.entropy_joint(np.array(4)) == 0)
        self.assertTrue(discrete.entropy_joint(np.array((4,))) == 0)
        self.assertTrue(discrete.entropy_joint(np.ones((1,))) == 0)
        self.assertTrue(discrete.entropy_joint(np.ones((1,1))) == 0)
        
        self.assertTrue(discrete.entropy_joint(np.array(((1.0, 2.0), (1.0, 2.0))))==1)
        self.assertTrue(discrete.entropy_joint(np.array(((1.0, 2.0), (1.0, 2.0))).T)==0)
        
        self.assertTrue(discrete.entropy_joint(np.array(1)).shape == tuple())
        self.assertTrue(discrete.entropy_joint(np.array((1,))).shape == tuple())
        self.assertTrue(discrete.entropy_joint(np.ones((1,1))).shape == tuple())        
        
        X = np.arange(3*4*5).reshape((3,4,5))
        self.assertTrue(isinstance(discrete.entropy_joint(X), np.float))
        self.assertTrue(discrete.entropy_joint(X).size == 1)
        self.assertTrue(np.allclose(discrete.entropy_joint(X), np.log2(5)))
        X = np.concatenate((X, X), axis=0)
        self.assertTrue(np.allclose(discrete.entropy_joint(X), np.log2(5)))

        np.random.seed(4759)
        X = np.random.randint(8, size=(1,10**4))
        self.assertTrue(discrete.entropy_joint(X) == discrete.entropy(X))
        
        np.random.seed(4759)
        X = np.random.randint(8, size=(3,10**5))
        self.assertTrue(np.all(np.abs(discrete.entropy_joint(X) - 3*discrete.entropy(X)) < 0.01))
        
        np.random.seed(4759)
        X = np.random.randint(8, size=(2,10**4))
        self.assertTrue(np.all(np.abs(discrete.entropy_joint(X) - 6.0) < 1E-02))
                
        np.random.seed(4759)
        X1 = np.random.randn(10**6)
        X2 = X1 * 0.5;
        Bins1 = np.linspace(-4,4,10000)
        P1 = np.digitize(X1, Bins1)
        P2 = np.digitize(X2, Bins1)
        H1 = discrete.entropy_joint(P1, base=2)
        H2 = discrete.entropy_joint(P2, base=2)
        self.assertTrue(abs(H1 - 1 - H2) < 1E-02)
        
        #Exception tests
        with self.assertRaises(ValueError):
            discrete.entropy_joint(np.array(()))
        with self.assertRaises(ValueError):
            discrete.entropy_joint(np.array((np.nan, 1)))
        with self.assertRaises(ValueError):
            discrete.entropy_joint(np.array((1,2)), Alphabet_X=(1,2,np.nan))
        with self.assertRaises(ValueError):
            discrete.entropy_joint(np.array((1,2)), fill_value=np.nan)
        with self.assertRaises(ValueError):
            discrete.entropy_joint(np.array((2, 1)), base=-1)
            
        #
        # Tests using missing data
        #
        #Immutability test
        X1 = np.random.randint(16, size=(10,10)) * 1.0
        X1[0,0] = -1
        X2 = np.copy(X1)
        discrete.entropy_joint(X2, fill_value=-1)
        self.assertTrue(np.all(X2 == X1))
        
        # Tests using missing data involving masked arrays
        X1 = np.random.randint(16, size=(10,10)) * 1.0
        X1[0,0] = -1
        Mask = np.zeros_like(X1)
        Mask[0,0] = 1
        self.assertTrue(discrete.entropy_joint(X1, fill_value=-1) == discrete.entropy_joint(np.ma.array(X1,mask=Mask)))        

        #Tests involving masked arrays        
        self.assertTrue(discrete.entropy_joint(np.ma.array(((4,4,2,2,2),(4,4,2,2,2)), mask=((0,0,1,0,0),(0,0,0,0,0)))) == 1)
        self.assertTrue(discrete.entropy_joint(np.ma.array(((5,1),(4,1)), mask=((0,0),(0,0)))) == 1)        
        self.assertTrue(discrete.entropy_joint(np.ma.array(((4.1,1.0,2.0),(4.1,1.0,2.0)), mask=((0,1,0),(0,0,0)))) == 1)
        self.assertTrue(discrete.entropy_joint(np.ma.array((('a','b','c'),('a','b','c')), mask=((0,1,0),(0,0,0)))) == 1)
        self.assertTrue(discrete.entropy_joint(np.ma.array((('N','b','c'),('N','b','c')), mask=((0,0,0),(0,1,0)))) == 1)
        self.assertTrue(discrete.entropy_joint(np.ma.array((('N/','bb','cc'),('N/','bb','cc')), mask=((0,1,0),(0,0,0)))) == 1)
        self.assertTrue(discrete.entropy_joint(np.ma.array((('N/A.','two','three'),('N/A.','two','three')), mask=((0,1,0),(0,0,0)))) == 1)
        self.assertTrue(np.isnan(discrete.entropy_joint(np.ma.array(((4.1,1.0,2.0),(4.1,1.0,2.0)), mask=((1,1,1),(0,0,0))))))
        
        #Tests involving masked arrays        
        self.assertTrue(discrete.entropy_joint(np.array(((4,4,3,2,2),(4,4,2,2,2))), fill_value=3) == 1)
        self.assertTrue(discrete.entropy_joint(np.array(((5,1),(4,1))), fill_value=-1) == 1)        
        self.assertTrue(discrete.entropy_joint(np.array(((4.1,1.0,2.0),(4.1,1.0,2.0))), fill_value=1.0) == 1)
        self.assertTrue(discrete.entropy_joint(np.array((('a','b','c'),('a','b','c'))), fill_value='b') == 1)
        self.assertTrue(discrete.entropy_joint(np.array((('N','b','c'),('N','N/A','c'))), fill_value='N/A') == 1)
        self.assertTrue(discrete.entropy_joint(np.array((('N/','bb','cc'),('N/','N/A','cc'))), fill_value='N/A') == 1)
        self.assertTrue(discrete.entropy_joint(np.array((('N/A.','two','three'),('N/A.','N/A','three'))), fill_value='N/A') == 1)
        self.assertTrue(np.isnan(discrete.entropy_joint(np.array(((-1,-1,-1),(-1,-1,-1))), fill_value=-1)))
        
        ### Tests involving Alphabet and non-ML estimation
        ###
        ###
        #Alphabet immutability 
        X1 = np.random.randint(16, size=(10,10))
        X2 = np.copy(X1)
        discrete.entropy_joint(X2, estimator=1, Alphabet_X=X2)
        self.assertTrue(np.all(X2 == X1))
        #No explicit alphabet
        self.assertTrue(discrete.entropy_joint(np.array(((1,2,1,2),(2,2,2,2))), estimator=1, Alphabet_X=None) == 1)
        #Larger alphabet (Counts 2 2; 7 extra bins)
        self.assertTrue(np.abs(discrete.entropy_joint(np.array(((1,2),(2,2))), estimator=1, Alphabet_X=np.array(((1,2,3),(1,2,3))), base=3)-1.9532)< 1E-03)
        #Empty alphabet
        with self.assertRaises(ValueError):
            discrete.entropy_joint([1, 2], estimator=1, Alphabet_X=np.array(()), base=3)
        #Smaller alphabet
        with self.assertRaises(ValueError):
            discrete.entropy_joint(np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array(((1,2),(1,None))), base=3)
        #Alphabet with incorrect dimensions
        with self.assertRaises(ValueError):
            discrete.entropy_joint(np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array((1,2)), base=3)        

    def test_entropy_cross(self):
        #Immutability test
        X1 = np.random.randint(16, size=(10,10))
        X1_copy = np.copy(X1)
        X2 = np.random.randint(16, size=(10,10))
        X2_copy = np.copy(X2)        
        discrete.entropy_cross(X1,X2)
        self.assertTrue(np.all(X1_copy == X1) and np.all(X2_copy == X2))
        
        self.assertTrue(discrete.entropy_cross(np.array((1.0, 1.0, 1.0, 1.0)), np.array((0.5, 0.5, 1.0, 1.0)), base=2) == 1)
        self.assertTrue(discrete.entropy_cross(np.array((1.0, 1.0, 1.0, 1.0)), np.array((1.0, 1.0, 1.0, 1.0)), base=2) == 0)
        self.assertTrue(discrete.entropy_cross(np.array((0.5, 0.5, 1.0, 1.0)), np.array((0.5, 0.5, 1.0, 1.0)), base=2) == 1)        
        self.assertTrue(abs(discrete.entropy_cross(np.array((1.0, 2.0)), np.array((1.0, 2.0)), base=np.exp(1))-0.693) < 1E-03)
        self.assertTrue(abs(discrete.entropy_cross(np.array((1.0, 1.0)), np.array((1.0, 2.0)), base=np.exp(1))-0.693) < 1E-03)       
        #Tests using cartesian_product=True                
        self.assertTrue(discrete.entropy_cross(np.array((1.0, 1.0, 1.0, 1.0)), np.array((0.5, 0.5, 1.0, 1.0)), cartesian_product=True, base=2) == 1)
        self.assertTrue(discrete.entropy_cross(np.array((1.0, 1.0, 1.0, 1.0)), np.array((1.0, 1.0, 1.0, 1.0)), cartesian_product=True, base=2) == 0)
        self.assertTrue(discrete.entropy_cross(np.array((0.5, 0.5, 1.0, 1.0)), np.array((0.5, 0.5, 1.0, 1.0)), cartesian_product=True, base=2) == 1)        
        self.assertTrue(abs(discrete.entropy_cross(np.array((1.0, 2.0)), np.array((1.0, 2.0)), cartesian_product=True, base=np.exp(1))-0.693) < 1E-03)
        self.assertTrue(abs(discrete.entropy_cross(np.array((1.0, 1.0)), np.array((1.0, 2.0)), cartesian_product=True, base=np.exp(1))-0.693) < 1E-03)               
        
        self.assertTrue(discrete.entropy_cross(np.ones(5),np.ones(5))==0)
        self.assertTrue(discrete.entropy_cross(np.ones((5,1)).T,np.ones((5,1)).T)==0)
        self.assertTrue(discrete.entropy_cross(np.ones((5,1)),np.ones((5,1))).size==5)
        self.assertTrue(np.all(discrete.entropy_cross(np.ones((5,1)),np.ones((5,1)))==0))
        #Tests using cartesian_product=True        
        self.assertTrue(discrete.entropy_cross(np.ones(5),np.ones(5), cartesian_product=True)==0)
        self.assertTrue(discrete.entropy_cross(np.ones((5,1)).T,np.ones((5,1)).T, cartesian_product=True)==0)
        self.assertTrue(discrete.entropy_cross(np.ones((5,1)),np.ones((5,1)), cartesian_product=True).size==25)
        self.assertTrue(np.all(discrete.entropy_cross(np.ones((5,1)),np.ones((5,1)), cartesian_product=True)==0))        
        
        self.assertTrue(discrete.entropy_cross(np.array(4),np.array(4)) == 0)
        self.assertTrue(discrete.entropy_cross(np.array((4,)),np.array((4,))) == 0)
        self.assertTrue(discrete.entropy_cross(np.ones((1,)),np.ones((1,))) == 0)
        self.assertTrue(discrete.entropy_cross(np.ones((1,1)),np.ones((1,1))) == 0)        
        #Tests using cartesian_product=True        
        self.assertTrue(discrete.entropy_cross(np.array(4),np.array(4), cartesian_product=True) == 0)
        self.assertTrue(discrete.entropy_cross(np.array((4,)),np.array((4,)), cartesian_product=True) == 0)
        self.assertTrue(discrete.entropy_cross(np.ones((1,)),np.ones((1,)), cartesian_product=True) == 0)
        self.assertTrue(discrete.entropy_cross(np.ones((1,1)),np.ones((1,1)), cartesian_product=True) == 0)        
        
        self.assertTrue(np.all(discrete.entropy_cross(np.array(((1.0, 2.0), (1.0, 2.0))))==1))
        self.assertTrue(np.all(discrete.entropy_cross(np.array(((1.0, 2.0), (1.0, 2.0))).T,np.array(((1.0, 2.0), (1.0, 2.0))).T)==0))
        #Tests using cartesian_product=True        
        self.assertTrue(np.all(discrete.entropy_cross(np.array(((1.0, 2.0), (1.0, 2.0))), cartesian_product=True)==1))
        self.assertTrue(np.all(discrete.entropy_cross(np.array(((1.0, 2.0), (1.0, 2.0))).T,np.array(((1.0, 2.0), (1.0, 2.0))).T, cartesian_product=True)==np.array(((0.0, np.inf),(np.inf, 0.0)))))
        
        self.assertTrue(discrete.entropy_cross(np.array(1),np.array(1)).shape == tuple())
        self.assertTrue(discrete.entropy_cross(np.array((1,)),np.array((1,))).shape == tuple())
        self.assertTrue(discrete.entropy_cross(np.ones((1,)),np.ones((1,))).shape == tuple())
        self.assertTrue(discrete.entropy_cross(np.ones((1,1)),np.ones((1,1))).shape == (1,))
        #Tests using cartesian_product=True
        self.assertTrue(discrete.entropy_cross(np.array(1),np.array(1),True).shape == tuple())
        self.assertTrue(discrete.entropy_cross(np.array((1,)),np.array((1,)),True).shape == tuple())
        self.assertTrue(discrete.entropy_cross(np.ones((1,)),np.ones((1,)),True).shape == tuple())
        self.assertTrue(discrete.entropy_cross(np.ones((1,1)),np.ones((1,1)),True).shape == (1,1))
        self.assertTrue(discrete.entropy_cross(np.ones((1,)),np.ones((1,1)),True).shape == (1,))
        self.assertTrue(discrete.entropy_cross(np.ones((1,1)),np.ones((1,)),True).shape == (1,))
        self.assertTrue(discrete.entropy_cross(np.ones((1,1)),np.ones((1,1)),True).shape == (1,1))        
        
        X = np.arange(3*4*5).reshape((3,4,5))
        self.assertTrue(isinstance(discrete.entropy_cross(X,X), np.ndarray))
        self.assertTrue(discrete.entropy_cross(X,X).shape == (3,4))
        self.assertTrue(np.allclose(discrete.entropy_cross(X,X), -np.log2(1.0/5)))
        #Tests using cartesian_product=True  
        X = np.arange(3*4*5).reshape((3,4,5))
        self.assertTrue(isinstance(discrete.entropy_cross(X,X, cartesian_product=True), np.ndarray))
        self.assertTrue(discrete.entropy_cross(X,X, cartesian_product=True).shape == (3,4,3,4))
        H = discrete.entropy_cross(X,X, cartesian_product=True)        
        self.assertTrue(np.allclose([H[i,j,i,j] for j in range(4) for i in range(3)], -np.log2(1.0/5)))        
        
        np.random.seed(4759)
        X = np.random.randint(16, size=(10,10**4))
        self.assertTrue(np.all(np.abs(discrete.entropy_cross(X,X) - 4.0) < 1E-02))
        #Tests using cartesian_product=True
        self.assertTrue(np.all(np.abs(discrete.entropy_cross(X,X, cartesian_product=True) - 4.0) < 1E-02))        
        
        np.random.seed(4759)
        X1 = np.random.randn(10**6)
        X2 = X1 * 0.5;
        Bins1 = np.linspace(-4,4,10000)
        P1 = np.digitize(X1, Bins1)
        P2 = np.digitize(X2, Bins1)
        H1 = discrete.entropy_cross(P1, base=2)
        H2 = discrete.entropy_cross(P2, base=2)
        self.assertTrue(abs(H1 - 1 - H2) < 1E-02)
        
        H1 = discrete.entropy_cross(P1, P1, base=2)
        H2 = discrete.entropy_cross(P2, P2, base=2)
        self.assertTrue(abs(H1 - 1 - H2) < 1E-02)
        
        #NB: More distribution tests implemented for divergence_kullbackleibler()
        
        with self.assertRaises(ValueError):
            discrete.entropy_cross(np.array(()), np.array((1,2,3)))
        with self.assertRaises(ValueError):
            discrete.entropy_cross(np.array((1,2,3)), np.array(()))            
        with self.assertRaises(ValueError):
            discrete.entropy_cross(np.array((np.nan, 1,2,4)), np.array((1,2,3)))
        with self.assertRaises(ValueError):
            discrete.entropy_cross(np.array((1,1,2,4)), np.array((1,2,3,np.nan)))
        with self.assertRaises(ValueError):
            discrete.entropy_cross(np.array((1,1,2,4)), np.array((1,1,2,4)), Alphabet_X=(1,1,2,4,np.nan), Alphabet_Y=(1,1,2,4))
        with self.assertRaises(ValueError):
            discrete.entropy_cross(np.array((1,1,2,4)), np.array((1,1,2,4)), Alphabet_X=(1,1,2,4), Alphabet_Y=(1,1,2,4,np.nan))
        with self.assertRaises(ValueError):
            discrete.entropy_cross(np.array((1,1,2,4)), np.array((1,1,2,4)), fill_value=np.nan)
        with self.assertRaises(ValueError):
            discrete.entropy_cross(np.array(((1,2,3),(4,5,6))), np.array((4,1,2,3)))
        try:
            discrete.entropy_cross(np.array(((1,2,3),(4,5,6))), np.array((4,1,2,3)), True)
            discrete.entropy_cross(np.array((1,2,3)), np.array((4,5,2,3)))
        except ValueError:
            self.fail("raised ValueError unexpectedly")
        with self.assertRaises(ValueError):
            discrete.entropy_cross(np.array((2, 1)), base=-1)
            
        #
        # Tests using missing data
        #
        #Immutability test        
        X1 = np.random.randint(16, size=(10,10)) * 1.0
        X1[0,0] = -1
        X1_copy = np.copy(X1)
        X2 = np.random.randint(16, size=(10,10))
        X2[0,1] = -1
        X2_copy = np.copy(X2)
        discrete.entropy_cross(X1,X2, fill_value=-1)
        self.assertTrue(np.all(X1_copy == X1) and np.all(X2_copy == X2))
        
        # Tests using missing data involving masked arrays
        X1 = np.random.randint(16, size=(10,10)) * 1.0
        X1[0,0] = -1
        Mask1 = np.zeros_like(X1)
        Mask1[0,0] = 1
        X2 = np.random.randint(16, size=(10,10)) * 1.0
        X2[0,1] = -1
        Mask2 = np.zeros_like(X2)
        Mask2[0,1] = 1
        self.assertTrue(np.all(discrete.entropy_cross(X1, X2, fill_value=-1) == discrete.entropy_cross(np.ma.array(X1,mask=Mask1), X2, fill_value=-1)))
        self.assertTrue(np.all(discrete.entropy_cross(X1, X2, fill_value=-1) == discrete.entropy_cross(X1, np.ma.array(X2,mask=Mask2), fill_value=-1)))
        self.assertTrue(np.all(discrete.entropy_cross(X1, X2, fill_value=-1) == discrete.entropy_cross(np.ma.array(X1,mask=Mask1), np.ma.array(X2,mask=Mask2), fill_value=-1)))        

        #Tests involving masked arrays
        self.assertTrue(discrete.entropy_cross(np.ma.array(((4,4,2,2,2)), mask=((0,0,1,1,1))), np.ma.array(((4,4,2,4,2)), mask=((0,0,1,1,1)))) == 0)
        self.assertTrue(np.isnan(discrete.entropy_cross(np.ma.array(((4,4,2,2,2)), mask=((1,1,1,1,1))), np.ma.array(((4,4,2,4,2)), mask=((0,0,1,1,1))))))
        self.assertTrue(np.isnan(discrete.entropy_cross(np.ma.array(((4,4,2,2,2)), mask=((0,0,1,1,1))), np.ma.array(((4,4,2,4,2)), mask=((1,1,1,1,1))))))

        self.assertTrue(discrete.entropy_cross(np.array(((4,4,-1,-1,-1))), np.array(((4,4,-1,-1,-1))), fill_value=-1) == 0)
        self.assertTrue(np.isnan(discrete.entropy_cross(np.array(((-1,-1,-1,-1,-1))), np.array(((4,4,-1,-1,-1))), fill_value=-1)))
        self.assertTrue(np.isnan(discrete.entropy_cross(np.array(((4,4,-1,-1,-1))), np.array(((-1,-1,-1,-1,-1))), fill_value=-1)))
                
        ### Tests involving Alphabet and non-ML estimation
        ###
        ###
        #Alphabet immutability 
        X1 = np.random.randint(16, size=(10,10))
        X2 = np.copy(X1)
        discrete.entropy_cross(X2, X2, estimator=1, Alphabet_X=X2, Alphabet_Y=X2)        
        self.assertTrue(np.all(X2 == X1))
        discrete.entropy_cross(X2, estimator=1, Alphabet_X=X2)
        self.assertTrue(np.all(X2 == X1))        
        #No explicit alphabet
        self.assertTrue(discrete.entropy_cross((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=None, Alphabet_Y=None) == discrete.entropy_cross((1,2,1,2,1,2),(2,2,2,2,1,1), Alphabet_X=None, Alphabet_Y=None))
        self.assertTrue(discrete.entropy_cross((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=None, Alphabet_Y=None) == discrete.entropy_cross((1,2,1,2,1,2),(2,2,2,2,1,1), Alphabet_X=None, Alphabet_Y=None))
        self.assertTrue(discrete.entropy_cross((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=None, Alphabet_Y=None) == discrete.entropy_cross((1,2,1,2,1,2),(2,2,2,2,1,1), Alphabet_X=None, Alphabet_Y=None))
        #No explicit alphabet -- Cartesian product
        A = discrete.entropy_cross(np.array(((1,2,1,2),(2,2,2,1))), estimator=1, Alphabet_X=None, base=3)
        B = discrete.entropy_cross(np.array(((1,2,1,2,1,2),(2,2,2,1,1,2))), estimator=0, Alphabet_X=None, base=3)        
        self.assertTrue(np.all(A==B))
        #Larger alphabet
        self.assertTrue(discrete.entropy_cross((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=(1,2,3), Alphabet_Y=None) == discrete.entropy_cross((1,2,1,2,1,2,3),(2,2,2,2,1,1), Alphabet_X=None, Alphabet_Y=None))
        self.assertTrue(discrete.entropy_cross((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=None, Alphabet_Y=(1,2,3)) == discrete.entropy_cross((1,2,1,2,1,2),(2,2,2,2,1,1,3), Alphabet_X=None, Alphabet_Y=None))
        self.assertTrue(discrete.entropy_cross((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=(1,2,3), Alphabet_Y=(1,2,3)) == discrete.entropy_cross((1,2,1,2,1,2,3),(2,2,2,2,1,1,3), Alphabet_X=None, Alphabet_Y=None))        
        #Larger alphabet -- Cartesian product
        A = discrete.entropy_cross(np.array(((1,2),(2,2))), estimator=1, Alphabet_X=np.array(((1,2,3),(1,2,3))), base=3)
        B = discrete.entropy_cross(np.array(((1,2,1,2,3),(2,2,2,3,1))), estimator=0, Alphabet_X=None, base=3)
        self.assertTrue(np.all(A==B))
        #Empty alphabet
        with self.assertRaises(ValueError):
            discrete.entropy_cross([1, 2], [1, 2], estimator=1, Alphabet_X=np.array(()), Alphabet_Y=None, base=3)
        with self.assertRaises(ValueError):
            discrete.entropy_cross([1, 2], [1, 2], estimator=1, Alphabet_X=None, Alphabet_Y=np.array(()), base=3)
        with self.assertRaises(ValueError):
            discrete.entropy_cross([1, 2], estimator=1, Alphabet_X=np.array(()), base=3)            
        #Smaller alphabet
        with self.assertRaises(ValueError):
            discrete.entropy_cross([1,2], [1,2], estimator=1, Alphabet_X=[1,2], Alphabet_Y=[1,1], base=3)
        with self.assertRaises(ValueError):
            discrete.entropy_cross([1,2], [1,2], estimator=1, Alphabet_X=[1,1], Alphabet_Y=[1,2], base=3)        
        with self.assertRaises(ValueError):
            discrete.entropy_cross(np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array(((1,2),(1,None))), base=3)
        #Alphabet with incorrect dimensions
        with self.assertRaises(ValueError):
            discrete.entropy_cross(np.array(((1,2),(1,2))), np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array((1,2)), Alphabet_Y=np.array(((1,2),(1,2))), base=3)
        with self.assertRaises(ValueError):
            discrete.entropy_cross(np.array(((1,2),(1,2))), np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array(((1,2),(1,2))), Alphabet_Y=np.array((1,2)), base=3)            
        with self.assertRaises(ValueError):
            discrete.entropy_cross(np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array((1,2)), base=3)            
            
    def test_divergence_kullbackleibler(self):
        #Immutability test
        X1 = np.random.randint(16, size=(10,10))
        X1_copy = np.copy(X1)
        X2 = np.random.randint(16, size=(10,10))
        X2_copy = np.copy(X2)
        discrete.divergence_kullbackleibler(X1,X2)
        self.assertTrue(np.all(X1_copy == X1) and np.all(X2_copy == X2))
        
        self.assertTrue(discrete.divergence_kullbackleibler(np.array((1.0, 1.0, 1.0, 1.0)), np.array((0.5, 0.5, 1.0, 1.0)), base=2) == 1)
        self.assertTrue(discrete.divergence_kullbackleibler(np.array((1.0, 1.0, 1.0, 1.0)), np.array((1.0, 1.0, 1.0, 1.0)), base=2) == 0)
        self.assertTrue(discrete.divergence_kullbackleibler(np.array((0.5, 0.5, 1.0, 1.0)), np.array((0.5, 0.5, 1.0, 1.0)), base=2) == 0)        
        self.assertTrue(abs(discrete.divergence_kullbackleibler(np.array((2.0, 2.0)), np.array((1.0, 2.0)), base=np.exp(1))-0.693) < 1E-03)
        self.assertTrue(abs(discrete.divergence_kullbackleibler(np.array((1.0, 1.0)), np.array((1.0, 2.0)), base=np.exp(1))-0.693) < 1E-03)       
        #Tests using cartesian_product=True                
        self.assertTrue(discrete.divergence_kullbackleibler(np.array((1.0, 1.0, 1.0, 1.0)), np.array((0.5, 0.5, 1.0, 1.0)), cartesian_product=True, base=2) == 1)
        self.assertTrue(discrete.divergence_kullbackleibler(np.array((1.0, 1.0, 1.0, 1.0)), np.array((1.0, 1.0, 1.0, 1.0)), cartesian_product=True, base=2) == 0)
        self.assertTrue(discrete.divergence_kullbackleibler(np.array((0.5, 0.5, 1.0, 1.0)), np.array((0.5, 0.5, 1.0, 1.0)), cartesian_product=True, base=2) == 0)        
        self.assertTrue(abs(discrete.divergence_kullbackleibler(np.array((2.0, 2.0)), np.array((1.0, 2.0)), cartesian_product=True, base=np.exp(1))-0.693) < 1E-03)
        self.assertTrue(abs(discrete.divergence_kullbackleibler(np.array((1.0, 1.0)), np.array((1.0, 2.0)), cartesian_product=True, base=np.exp(1))-0.693) < 1E-03)               
        
        self.assertTrue(discrete.divergence_kullbackleibler(np.ones(5),np.ones(5))==0)
        self.assertTrue(discrete.divergence_kullbackleibler(np.ones((5,1)).T,np.ones((5,1)).T)==0)
        self.assertTrue(discrete.divergence_kullbackleibler(np.ones((5,1)),np.ones((5,1))).size==5)
        self.assertTrue(np.all(discrete.divergence_kullbackleibler(np.ones((5,1)),np.ones((5,1)))==0))
        #Tests using cartesian_product=True        
        self.assertTrue(discrete.divergence_kullbackleibler(np.ones(5),np.ones(5), cartesian_product=True)==0)
        self.assertTrue(discrete.divergence_kullbackleibler(np.ones((5,1)).T,np.ones((5,1)).T, cartesian_product=True)==0)
        self.assertTrue(discrete.divergence_kullbackleibler(np.ones((5,1)),np.ones((5,1)), cartesian_product=True).size==25)
        self.assertTrue(np.all(discrete.divergence_kullbackleibler(np.ones((5,1)),np.ones((5,1)), cartesian_product=True)==0))        
        
        self.assertTrue(discrete.divergence_kullbackleibler(np.array(4),np.array(4)) == 0)
        self.assertTrue(discrete.divergence_kullbackleibler(np.array((4,)),np.array((4,))) == 0)
        self.assertTrue(discrete.divergence_kullbackleibler(np.ones((1,)),np.ones((1,))) == 0)
        self.assertTrue(discrete.divergence_kullbackleibler(np.ones((1,1)),np.ones((1,1))) == 0)        
        #Tests using cartesian_product=True        
        self.assertTrue(discrete.divergence_kullbackleibler(np.array(4),np.array(4), cartesian_product=True) == 0)
        self.assertTrue(discrete.divergence_kullbackleibler(np.array((4,)),np.array((4,)), cartesian_product=True) == 0)
        self.assertTrue(discrete.divergence_kullbackleibler(np.ones((1,)),np.ones((1,)), cartesian_product=True) == 0)
        self.assertTrue(discrete.divergence_kullbackleibler(np.ones((1,1)),np.ones((1,1)), cartesian_product=True) == 0)        
        
        self.assertTrue(np.all(discrete.divergence_kullbackleibler(np.array(((1.0, 2.0), (1.0, 2.0))))==0))
        self.assertTrue(np.all(discrete.divergence_kullbackleibler(np.array(((1.0, 2.0), (1.0, 2.0))).T,np.array(((1.0, 2.0), (1.0, 2.0))).T)==0))
        #Tests using cartesian_product=True        
        self.assertTrue(np.all(discrete.divergence_kullbackleibler(np.array(((1.0, 2.0), (1.0, 2.0))), cartesian_product=True)==0))
        self.assertTrue(np.all(discrete.divergence_kullbackleibler(np.array(((1.0, 2.0), (1.0, 2.0))).T,np.array(((1.0, 2.0), (1.0, 2.0))).T, cartesian_product=True)==np.array(((0.0, np.inf),(np.inf, 0.0)))))
        
        self.assertTrue(discrete.divergence_kullbackleibler(np.array(1),np.array(1)).shape == tuple())
        self.assertTrue(discrete.divergence_kullbackleibler(np.array((1,)),np.array((1,))).shape == tuple())
        self.assertTrue(discrete.divergence_kullbackleibler(np.ones((1,)),np.ones((1,))).shape == tuple())
        self.assertTrue(discrete.divergence_kullbackleibler(np.ones((1,1)),np.ones((1,1))).shape == (1,))
        #Tests using cartesian_product=True
        self.assertTrue(discrete.divergence_kullbackleibler(np.array(1),np.array(1),True).shape == tuple())
        self.assertTrue(discrete.divergence_kullbackleibler(np.array((1,)),np.array((1,)),True).shape == tuple())
        self.assertTrue(discrete.divergence_kullbackleibler(np.ones((1,)),np.ones((1,)),True).shape == tuple())
        self.assertTrue(discrete.divergence_kullbackleibler(np.ones((1,1)),np.ones((1,1)),True).shape == (1,1))
        self.assertTrue(discrete.divergence_kullbackleibler(np.ones((1,)),np.ones((1,1)),True).shape == (1,))
        self.assertTrue(discrete.divergence_kullbackleibler(np.ones((1,1)),np.ones((1,)),True).shape == (1,))
        self.assertTrue(discrete.divergence_kullbackleibler(np.ones((1,1)),np.ones((1,1)),True).shape == (1,1))
        
        X = np.arange(3*4*5).reshape((3,4,5))
        self.assertTrue(isinstance(discrete.divergence_kullbackleibler(X,X), np.ndarray))
        self.assertTrue(discrete.divergence_kullbackleibler(X,X).shape == (3,4))
        self.assertTrue(np.all(discrete.divergence_kullbackleibler(X,X)==0))
        #Tests using cartesian_product=True  
        X = np.arange(3*4*5).reshape((3,4,5))
        self.assertTrue(isinstance(discrete.divergence_kullbackleibler(X,X, cartesian_product=True), np.ndarray))
        self.assertTrue(discrete.divergence_kullbackleibler(X,X, cartesian_product=True).shape == (3,4,3,4))
        H = discrete.divergence_kullbackleibler(X,X, cartesian_product=True)        
        self.assertTrue(np.all(np.array([H[i,j,i,j] for j in range(4) for i in range(3)]) == 0))
        
        np.random.seed(4759)
        X1 = np.random.randn(10**6)
        X2 = X1 * 0.5;
        Bins1 = np.linspace(-4,4,10000)
        P1 = np.digitize(X1, Bins1)
        P2 = np.digitize(X2, Bins1)
        H = discrete.divergence_kullbackleibler(P2, P1, base=np.exp(1))
        self.assertTrue(np.abs(H - 0.3181) < 1E-2)        
        
        Scales = np.linspace(0.1,1.0,8).reshape(2,2,2)
        P2 = np.empty((2,2,2,10**6))
        P1 = np.append(P1,np.arange(10001))
        for i in np.arange(Scales.shape[0]):
            for j in np.arange(Scales.shape[1]):
                for k in np.arange(Scales.shape[2]):
                    X2 = X1 * Scales[i,j,k]
                    P2[i,j,k,:] = np.digitize(X2, Bins1)
        H = discrete.divergence_kullbackleibler(P2, P1, True, base=np.exp(1))
        Scales = Scales.reshape(-1)
        H = H.reshape(-1)
        H_predicted = np.log(1.0/Scales) + (Scales**2 / 2) - 0.5
        self.assertTrue(np.all(np.abs(H - H_predicted) < 2*1E-2))             
        
        #
        # Tests using missing data (basic)
        #
        #Immutability test        
        X1 = np.random.randint(16, size=(10,10)) * 1.0
        X1[0,0] = -1
        X1_copy = np.copy(X1)
        X2 = np.random.randint(16, size=(10,10))
        X2[0,1] = -1
        X2_copy = np.copy(X2)
        discrete.divergence_kullbackleibler(X1,X2, fill_value=-1)
        self.assertTrue(np.all(X1_copy == X1) and np.all(X2_copy == X2))
#        self.assertFalse(np.all(discrete.divergence_kullbackleibler(X1,X2, fill_value=-1) == discrete.divergence_kullbackleibler(X1,X2, fill_value=None)))

        ### Tests involving Alphabet and non-ML estimation
        ###
        ###
        #Alphabet immutability 
        X1 = np.random.randint(16, size=(10,10))
        X2 = np.copy(X1)
        discrete.divergence_kullbackleibler(X2, X2, estimator=1, Alphabet_X=X2, Alphabet_Y=X2)        
        self.assertTrue(np.all(X2 == X1))
        discrete.divergence_kullbackleibler(X2, estimator=1, Alphabet_X=X2)
        self.assertTrue(np.all(X2 == X1))        
        #No explicit alphabet
        self.assertTrue(discrete.divergence_kullbackleibler((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=None, Alphabet_Y=None) == discrete.divergence_kullbackleibler((1,2,1,2,1,2),(2,2,2,2,1,1), Alphabet_X=None, Alphabet_Y=None))
        self.assertTrue(discrete.divergence_kullbackleibler((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=None, Alphabet_Y=None) == discrete.divergence_kullbackleibler((1,2,1,2,1,2),(2,2,2,2,1,1), Alphabet_X=None, Alphabet_Y=None))
        self.assertTrue(discrete.divergence_kullbackleibler((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=None, Alphabet_Y=None) == discrete.divergence_kullbackleibler((1,2,1,2,1,2),(2,2,2,2,1,1), Alphabet_X=None, Alphabet_Y=None))
        #No explicit alphabet -- Cartesian product
        A = discrete.divergence_kullbackleibler(np.array(((1,2,1,2),(2,2,2,1))), estimator=1, Alphabet_X=None, base=3)
        B = discrete.divergence_kullbackleibler(np.array(((1,2,1,2,1,2),(2,2,2,1,1,2))), estimator=0, Alphabet_X=None, base=3)        
        self.assertTrue(np.all(A==B))
        #Larger alphabet
        self.assertTrue(discrete.divergence_kullbackleibler((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=(1,2,3), Alphabet_Y=None) == discrete.divergence_kullbackleibler((1,2,1,2,1,2,3),(2,2,2,2,1,1), Alphabet_X=None, Alphabet_Y=None))
        self.assertTrue(discrete.divergence_kullbackleibler((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=None, Alphabet_Y=(1,2,3)) == discrete.divergence_kullbackleibler((1,2,1,2,1,2),(2,2,2,2,1,1,3), Alphabet_X=None, Alphabet_Y=None))
        self.assertTrue(discrete.divergence_kullbackleibler((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=(1,2,3), Alphabet_Y=(1,2,3)) == discrete.divergence_kullbackleibler((1,2,1,2,1,2,3),(2,2,2,2,1,1,3), Alphabet_X=None, Alphabet_Y=None))        
        #Larger alphabet -- Cartesian product
        A = discrete.divergence_kullbackleibler(np.array(((1,2),(2,2))), estimator=1, Alphabet_X=np.array(((1,2,3),(1,2,3))), base=3)
        B = discrete.divergence_kullbackleibler(np.array(((1,2,1,2,3),(2,2,2,3,1))), estimator=0, Alphabet_X=None, base=3)
        self.assertTrue(np.allclose(A,B))
        #Empty alphabet
        with self.assertRaises(ValueError):
            discrete.divergence_kullbackleibler([1, 2], [1, 2], estimator=1, Alphabet_X=np.array(()), Alphabet_Y=None, base=3)
        with self.assertRaises(ValueError):
            discrete.divergence_kullbackleibler([1, 2], [1, 2], estimator=1, Alphabet_X=None, Alphabet_Y=np.array(()), base=3)
        with self.assertRaises(ValueError):
            discrete.divergence_kullbackleibler([1, 2], estimator=1, Alphabet_X=np.array(()), base=3)            
        #Smaller alphabet
        with self.assertRaises(ValueError):
            discrete.divergence_kullbackleibler([1,2], [1,2], estimator=1, Alphabet_X=[1,2], Alphabet_Y=[1,1], base=3)
        with self.assertRaises(ValueError):
            discrete.divergence_kullbackleibler([1,2], [1,2], estimator=1, Alphabet_X=[1,1], Alphabet_Y=[1,2], base=3)        
        with self.assertRaises(ValueError):
            discrete.divergence_kullbackleibler(np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array(((1,2),(1,None))), base=3)
        #Alphabet with incorrect dimensions
        with self.assertRaises(ValueError):
            discrete.divergence_kullbackleibler(np.array(((1,2),(1,2))), np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array((1,2)), Alphabet_Y=np.array(((1,2),(1,2))), base=3)
        with self.assertRaises(ValueError):
            discrete.divergence_kullbackleibler(np.array(((1,2),(1,2))), np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array(((1,2),(1,2))), Alphabet_Y=np.array((1,2)), base=3)            
        with self.assertRaises(ValueError):
            discrete.divergence_kullbackleibler(np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array((1,2)), base=3)
        
    def test_divergence_jensenshannon(self):
        #Immutability test
        X1 = np.random.randint(16, size=(10,10))
        X1_copy = np.copy(X1)
        X2 = np.random.randint(16, size=(10,10))
        X2_copy = np.copy(X2)        
        discrete.divergence_jensenshannon(X1,X2)
        self.assertTrue(np.all(X1_copy == X1) and np.all(X2_copy == X2))
        
        self.assertTrue(discrete.divergence_jensenshannon(np.array((1.0, 1.0, 1.0, 1.0)), np.array((2.0, 2.0, 2.0, 2.0)), base=2) == 1)
        self.assertTrue(discrete.divergence_jensenshannon(np.array((1.0, 1.0, 1.0, 1.0)), np.array((1.0, 1.0, 1.0, 1.0)), base=2) == 0)
        self.assertTrue(discrete.divergence_jensenshannon(np.array((0.5, 0.5, 1.0, 1.0)), np.array((0.5, 0.5, 1.0, 1.0)), base=2) == 0)        
        self.assertTrue(abs(discrete.divergence_jensenshannon(np.array((1.0, 1.0)), np.array((2.0, 2.0)), base=np.exp(1))-0.693) < 1E-03)
        #Tests using cartesian_product=True
        self.assertTrue(discrete.divergence_jensenshannon(np.array((1.0, 1.0, 1.0, 1.0)), np.array((2.0, 2.0, 2.0, 2.0)), cartesian_product=True, base=2) == 1)
        self.assertTrue(discrete.divergence_jensenshannon(np.array((1.0, 1.0, 1.0, 1.0)), np.array((1.0, 1.0, 1.0, 1.0)), cartesian_product=True, base=2) == 0)
        self.assertTrue(discrete.divergence_jensenshannon(np.array((0.5, 0.5, 1.0, 1.0)), np.array((0.5, 0.5, 1.0, 1.0)), cartesian_product=True, base=2) == 0)        
        self.assertTrue(abs(discrete.divergence_jensenshannon(np.array((1.0, 1.0)), np.array((2.0, 2.0)), cartesian_product=True, base=np.exp(1))-0.693) < 1E-03)        
        
        self.assertTrue(discrete.divergence_jensenshannon(np.ones(5),np.ones(5))==0)
        self.assertTrue(discrete.divergence_jensenshannon(np.ones((5,1)).T,np.ones((5,1)).T)==0)
        self.assertTrue(discrete.divergence_jensenshannon(np.ones((5,1)),np.ones((5,1))).size==5)
        self.assertTrue(np.all(discrete.divergence_jensenshannon(np.ones((5,1)),np.ones((5,1)))==0))
        #Tests using cartesian_product=True        
        self.assertTrue(discrete.divergence_jensenshannon(np.ones(5),np.ones(5), cartesian_product=True)==0)
        self.assertTrue(discrete.divergence_jensenshannon(np.ones((5,1)).T,np.ones((5,1)).T, cartesian_product=True)==0)
        self.assertTrue(discrete.divergence_jensenshannon(np.ones((5,1)),np.ones((5,1)), cartesian_product=True).size==25)
        self.assertTrue(np.all(discrete.divergence_jensenshannon(np.ones((5,1)),np.ones((5,1)), cartesian_product=True)==0))        
        
        self.assertTrue(discrete.divergence_jensenshannon(np.array(4),np.array(4)) == 0)
        self.assertTrue(discrete.divergence_jensenshannon(np.array((4,)),np.array((4,))) == 0)
        self.assertTrue(discrete.divergence_jensenshannon(np.ones((1,)),np.ones((1,))) == 0)
        self.assertTrue(discrete.divergence_jensenshannon(np.ones((1,1)),np.ones((1,1))) == 0)        
        #Tests using cartesian_product=True        
        self.assertTrue(discrete.divergence_jensenshannon(np.array(4),np.array(4), cartesian_product=True) == 0)
        self.assertTrue(discrete.divergence_jensenshannon(np.array((4,)),np.array((4,)), cartesian_product=True) == 0)
        self.assertTrue(discrete.divergence_jensenshannon(np.ones((1,)),np.ones((1,)), cartesian_product=True) == 0)
        self.assertTrue(discrete.divergence_jensenshannon(np.ones((1,1)),np.ones((1,1)), cartesian_product=True) == 0)        
        
        self.assertTrue(np.all(discrete.divergence_jensenshannon(np.array(((1.0, 2.0), (1.0, 2.0))))==0))
        self.assertTrue(np.all(discrete.divergence_jensenshannon(np.array(((1.0, 1.0), (2.0, 2.0))),np.array(((1.0, 1.0), (2.0, 2.0))))==0))
        #Tests using cartesian_product=True        
        self.assertTrue(np.all(discrete.divergence_jensenshannon(np.array(((1.0, 1.0), (2.0, 2.0))), cartesian_product=True)==np.array(((0.0, 1),(1, 0.0)))))
        self.assertTrue(np.all(discrete.divergence_jensenshannon(np.array(((1.0, 1.0), (2.0, 2.0))), np.array(((1.0, 1.0), (2.0, 2.0))), cartesian_product=True)==np.array(((0.0, 1),(1, 0.0)))))
        
        self.assertTrue(discrete.divergence_jensenshannon(np.array(1),np.array(1)).shape == tuple())
        self.assertTrue(discrete.divergence_jensenshannon(np.array((1,)),np.array((1,))).shape == tuple())
        self.assertTrue(discrete.divergence_jensenshannon(np.ones((1,)),np.ones((1,))).shape == tuple())
        self.assertTrue(discrete.divergence_jensenshannon(np.ones((1,1)),np.ones((1,1))).shape == (1,))
        #Tests using cartesian_product=True
        self.assertTrue(discrete.divergence_jensenshannon(np.array(1),np.array(1),True).shape == tuple())
        self.assertTrue(discrete.divergence_jensenshannon(np.array((1,)),np.array((1,)),True).shape == tuple())
        self.assertTrue(discrete.divergence_jensenshannon(np.ones((1,)),np.ones((1,)),True).shape == tuple())
        self.assertTrue(discrete.divergence_jensenshannon(np.ones((1,1)),np.ones((1,1)),True).shape == (1,1))
        self.assertTrue(discrete.divergence_jensenshannon(np.ones((1,)),np.ones((1,1)),True).shape == (1,))
        self.assertTrue(discrete.divergence_jensenshannon(np.ones((1,1)),np.ones((1,)),True).shape == (1,))
        self.assertTrue(discrete.divergence_jensenshannon(np.ones((1,1)),np.ones((1,1)),True).shape == (1,1))        
        
        #Test based on interpreting JSD as mutual information between X, sourced from mixture distribution, and Y, the mixture component indicator
        Bins1 = np.linspace(-5,5,10000)
        P1 = norm.pdf(Bins1,loc=-0.5)
        P2 = norm.pdf(Bins1,loc=1.0)
        P1 = P1 / np.sum(P1)
        P2 = P2 / np.sum(P2)
        P = np.append(P1, P2)
        P = P / np.sum(P)
        #Sample from component distributions
        I = np.random.choice(P.size, 10**6, p=P)
        #Get component indicator
        C = I >= 10000
        #Mod operation to complete mixture sampling
        I = I % 10000
        #Compute JSD based on component pmfs
        SampleP1 = np.random.choice(P1.size, size=10**6, p=P1)
        SampleP2 = np.random.choice(P2.size, size=10**6, p=P2)
        JSD = discrete.divergence_jensenshannon(SampleP1, SampleP2)
        #Estimate mutual information between C and I
        MI = discrete.entropy(C) - discrete.entropy_conditional(C, I)
        self.assertTrue(np.abs(JSD - MI) < 0.01)

        #Additional test based on Kullback-Leibler divergence
        M = 0.5 * (P1 + P2)
        JSD2 = 0.5 * (discrete.divergence_kullbackleibler_pmf(P1, M) + discrete.divergence_kullbackleibler_pmf(P2, M))
        self.assertTrue(np.abs(JSD - JSD2) < 0.01)
        
        with self.assertRaises(ValueError):
            discrete.divergence_jensenshannon(np.array(()), np.array((1,2,3)))
        with self.assertRaises(ValueError):
            discrete.divergence_jensenshannon(np.array((1,2,3)), np.array(()))            
        with self.assertRaises(ValueError):
            discrete.divergence_jensenshannon(np.array((np.nan, 1)), np.array((1,2,3)))
        with self.assertRaises(ValueError):
            discrete.divergence_jensenshannon(np.array((1,2,3)), np.array((np.nan,1,2,3)))
        with self.assertRaises(ValueError):
            discrete.divergence_jensenshannon(np.array((1,2,3)), np.array((1,2,3)), Alphabet_X=(1,2,3,np.nan), Alphabet_Y=(1,2,3))
        with self.assertRaises(ValueError):
            discrete.divergence_jensenshannon(np.array((1,2,3)), np.array((1,2,3)), Alphabet_X=(1,2,3), Alphabet_Y=(1,2,3,np.nan))
        with self.assertRaises(ValueError):
            discrete.divergence_jensenshannon(np.array((1,2,3)), np.array((1,2,3)), fill_value=np.nan)            
        with self.assertRaises(ValueError):
            discrete.divergence_jensenshannon(np.array(((1,2,3),(4,5,6))), np.array((4,1,2,3)))
        try:
            discrete.divergence_jensenshannon(np.array(((1,2,3),(4,5,6))), np.array((4,1,2,3)), True)
            discrete.divergence_jensenshannon(np.array((1,2,5)), np.array((4,1,2,3)))
        except ValueError:
            self.fail("raised ValueError unexpectedly")
        with self.assertRaises(ValueError):
            discrete.divergence_jensenshannon(np.array((2, 1)), base=-1)
            
        #
        # Tests using missing data
        #
        #Immutability test        
        X1 = np.random.randint(16, size=(10,10)) * 1.0
        X1[0,0] = -1
        X1_copy = np.copy(X1)
        X2 = np.random.randint(16, size=(10,10))
        X2[0,1] = -1
        X2_copy = np.copy(X2)
        discrete.divergence_jensenshannon(X1,X2, fill_value=-1)
        self.assertTrue(np.all(X1_copy == X1) and np.all(X2_copy == X2))
        
        # Tests using missing data involving masked arrays
        X1 = np.random.randint(16, size=(10,10)) * 1.0
        X1[0,0] = -1
        Mask1 = np.zeros_like(X1)
        Mask1[0,0] = 1
        X2 = np.random.randint(16, size=(10,10)) * 1.0
        X2[0,1] = -1
        Mask2 = np.zeros_like(X2)
        Mask2[0,1] = 1
        self.assertTrue(np.all(discrete.divergence_jensenshannon(X1, X2, fill_value=-1) == discrete.divergence_jensenshannon(np.ma.array(X1,mask=Mask1), X2, fill_value=-1)))
        self.assertTrue(np.all(discrete.divergence_jensenshannon(X1, X2, fill_value=-1) == discrete.divergence_jensenshannon(X1, np.ma.array(X2,mask=Mask2), fill_value=-1)))
        self.assertTrue(np.all(discrete.divergence_jensenshannon(X1, X2, fill_value=-1) == discrete.divergence_jensenshannon(np.ma.array(X1,mask=Mask1), np.ma.array(X2,mask=Mask2), fill_value=-1)))        

        #Tests involving masked arrays
        self.assertTrue(discrete.divergence_jensenshannon(np.ma.array(((4,4,2,2,2)), mask=((0,0,1,1,1))), np.ma.array(((4,4,2,4,2)), mask=((0,0,1,1,1)))) == 0)
        self.assertTrue(np.isnan(discrete.divergence_jensenshannon(np.ma.array(((4,4,2,2,2)), mask=((1,1,1,1,1))), np.ma.array(((4,4,2,4,2)), mask=((0,0,1,1,1))))))
        self.assertTrue(np.isnan(discrete.divergence_jensenshannon(np.ma.array(((4,4,2,2,2)), mask=((0,0,1,1,1))), np.ma.array(((4,4,2,4,2)), mask=((1,1,1,1,1))))))

        self.assertTrue(discrete.divergence_jensenshannon(np.array(((4,4,-1,-1,-1))), np.array(((4,4,-1,-1,-1))), fill_value=-1) == 0)
        self.assertTrue(np.isnan(discrete.divergence_jensenshannon(np.array(((-1,-1,-1,-1,-1))), np.array(((4,4,-1,-1,-1))), fill_value=-1)))
        self.assertTrue(np.isnan(discrete.divergence_jensenshannon(np.array(((4,4,-1,-1,-1))), np.array(((-1,-1,-1,-1,-1))), fill_value=-1)))
                    
        ### Tests involving Alphabet and non-ML estimation
        ###
        ###
        #Alphabet immutability 
        X1 = np.random.randint(16, size=(10,10))
        X2 = np.copy(X1)
        discrete.divergence_jensenshannon(X2, X2, estimator=1, Alphabet_X=X2, Alphabet_Y=X2)        
        self.assertTrue(np.all(X2 == X1))
        discrete.divergence_jensenshannon(X2, estimator=1, Alphabet_X=X2)
        self.assertTrue(np.all(X2 == X1))        
        #No explicit alphabet
        self.assertTrue(discrete.divergence_jensenshannon((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=None, Alphabet_Y=None) == discrete.divergence_jensenshannon((1,2,1,2,1,2),(2,2,2,2,1,1), Alphabet_X=None, Alphabet_Y=None))
        self.assertTrue(discrete.divergence_jensenshannon((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=None, Alphabet_Y=None) == discrete.divergence_jensenshannon((1,2,1,2,1,2),(2,2,2,2,1,1), Alphabet_X=None, Alphabet_Y=None))
        self.assertTrue(discrete.divergence_jensenshannon((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=None, Alphabet_Y=None) == discrete.divergence_jensenshannon((1,2,1,2,1,2),(2,2,2,2,1,1), Alphabet_X=None, Alphabet_Y=None))
        #No explicit alphabet -- Cartesian product
        A = discrete.divergence_jensenshannon(np.array(((1,2,1,2),(2,2,2,1))), estimator=1, Alphabet_X=None, base=3)
        B = discrete.divergence_jensenshannon(np.array(((1,2,1,2,1,2),(2,2,2,1,1,2))), estimator=0, Alphabet_X=None, base=3)        
        self.assertTrue(np.all(A==B))
        #Larger alphabet
        self.assertTrue(discrete.divergence_jensenshannon((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=(1,2,3), Alphabet_Y=None) == discrete.divergence_jensenshannon((1,2,1,2,1,2,3),(2,2,2,2,1,1), Alphabet_X=None, Alphabet_Y=None))
        self.assertTrue(discrete.divergence_jensenshannon((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=None, Alphabet_Y=(1,2,3)) == discrete.divergence_jensenshannon((1,2,1,2,1,2),(2,2,2,2,1,1,3), Alphabet_X=None, Alphabet_Y=None))
        self.assertTrue(discrete.divergence_jensenshannon((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=(1,2,3), Alphabet_Y=(1,2,3)) == discrete.divergence_jensenshannon((1,2,1,2,1,2,3),(2,2,2,2,1,1,3), Alphabet_X=None, Alphabet_Y=None))        
        #Larger alphabet -- Cartesian product
        A = discrete.divergence_jensenshannon(np.array(((1,2),(2,2))), estimator=1, Alphabet_X=np.array(((1,2,3),(1,2,3))), base=3)
        B = discrete.divergence_jensenshannon(np.array(((1,2,1,2,3),(2,2,2,3,1))), estimator=0, Alphabet_X=None, base=3)
        self.assertTrue(np.all(A==B))
        #Empty alphabet
        with self.assertRaises(ValueError):
            discrete.divergence_jensenshannon([1, 2], [1, 2], estimator=1, Alphabet_X=np.array(()), Alphabet_Y=None, base=3)
        with self.assertRaises(ValueError):
            discrete.divergence_jensenshannon([1, 2], [1, 2], estimator=1, Alphabet_X=None, Alphabet_Y=np.array(()), base=3)
        with self.assertRaises(ValueError):
            discrete.divergence_jensenshannon([1, 2], estimator=1, Alphabet_X=np.array(()), base=3)            
        #Smaller alphabet
        with self.assertRaises(ValueError):
            discrete.divergence_jensenshannon([1,2], [1,2], estimator=1, Alphabet_X=[1,2], Alphabet_Y=[1,1], base=3)
        with self.assertRaises(ValueError):
            discrete.divergence_jensenshannon([1,2], [1,2], estimator=1, Alphabet_X=[1,1], Alphabet_Y=[1,2], base=3)        
        with self.assertRaises(ValueError):
            discrete.divergence_jensenshannon(np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array(((1,2),(1,None))), base=3)
        #Alphabet with incorrect dimensions
        with self.assertRaises(ValueError):
            discrete.divergence_jensenshannon(np.array(((1,2),(1,2))), np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array((1,2)), Alphabet_Y=np.array(((1,2),(1,2))), base=3)
        with self.assertRaises(ValueError):
            discrete.divergence_jensenshannon(np.array(((1,2),(1,2))), np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array(((1,2),(1,2))), Alphabet_Y=np.array((1,2)), base=3)            
        with self.assertRaises(ValueError):
            discrete.divergence_jensenshannon(np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array((1,2)), base=3)            
                
    def test_divergence_kullbackleibler_symmetrised(self):
        #Immutability test
        X1 = np.random.randint(16, size=(10,10))
        X1_copy = np.copy(X1)
        X2 = np.random.randint(16, size=(10,10))
        X2_copy = np.copy(X2)
        discrete.divergence_kullbackleibler_symmetrised(X1,X2)
        self.assertTrue(np.all(X1_copy == X1) and np.all(X2_copy == X2))        
        
        self.assertTrue(discrete.divergence_kullbackleibler_symmetrised(np.array(1),np.array(1)).shape == tuple())
        self.assertTrue(discrete.divergence_kullbackleibler_symmetrised(np.array((1,)),np.array((1,))).shape == tuple())
        self.assertTrue(discrete.divergence_kullbackleibler_symmetrised(np.ones((1,)),np.ones((1,))).shape == tuple())
        self.assertTrue(discrete.divergence_kullbackleibler_symmetrised(np.ones((1,1)),np.ones((1,1))).shape == (1,))
        #Tests using cartesian_product=True
        self.assertTrue(discrete.divergence_kullbackleibler_symmetrised(np.array(1),np.array(1),True).shape == tuple())
        self.assertTrue(discrete.divergence_kullbackleibler_symmetrised(np.array((1,)),np.array((1,)),True).shape == tuple())
        self.assertTrue(discrete.divergence_kullbackleibler_symmetrised(np.ones((1,)),np.ones((1,)),True).shape == tuple())
        self.assertTrue(discrete.divergence_kullbackleibler_symmetrised(np.ones((1,1)),np.ones((1,1)),True).shape == (1,1))
        self.assertTrue(discrete.divergence_kullbackleibler_symmetrised(np.ones((1,)),np.ones((1,1)),True).shape == (1,))
        self.assertTrue(discrete.divergence_kullbackleibler_symmetrised(np.ones((1,1)),np.ones((1,)),True).shape == (1,))
        self.assertTrue(discrete.divergence_kullbackleibler_symmetrised(np.ones((1,1)),np.ones((1,1)),True).shape == (1,1))        

        self.assertTrue(np.allclose(discrete.divergence_kullbackleibler_symmetrised(np.array((1,2,1)), np.array((2,1,2))), 2.0/3))
        
        X = np.array(((1,2,1),(2,2,1),(1,1,2)))
        Y = np.array(((1,2,1),(1,2,2)))
        H = discrete.divergence_kullbackleibler_symmetrised(X,Y, True)
        self.assertTrue(np.allclose(H, np.array(((0.0,2.0/3),(2.0/3,0.0),(0.0,2.0/3)))))
        H = discrete.divergence_kullbackleibler_symmetrised(Y)
        self.assertTrue(np.allclose(H, np.array(((0.0,2.0/3),(2.0/3,0.0)))))
        
        #
        # Tests using missing data (basic)
        #
        #Immutability test        
        X1 = np.random.randint(16, size=(10,10)) * 1.0
        X1[0,0] = -1
        X1_copy = np.copy(X1)
        X2 = np.random.randint(16, size=(10,10))
        X2[0,1] = -1
        X2_copy = np.copy(X2)
        discrete.divergence_kullbackleibler_symmetrised(X1,X2, fill_value=-1)
        self.assertTrue(np.all(X1_copy == X1) and np.all(X2_copy == X2))
#        self.assertFalse(np.all(discrete.divergence_kullbackleibler_symmetrised(X1,X2, fill_value=-1) == discrete.divergence_kullbackleibler_symmetrised(X1,X2, fill_value=None)))

        ### Tests involving Alphabet and non-ML estimation
        ###
        ###
        #Alphabet immutability 
        X1 = np.random.randint(16, size=(10,10))
        X2 = np.copy(X1)
        discrete.divergence_kullbackleibler_symmetrised(X2, X2, estimator=1, Alphabet_X=X2, Alphabet_Y=X2)        
        self.assertTrue(np.all(X2 == X1))
        discrete.divergence_kullbackleibler_symmetrised(X2, estimator=1, Alphabet_X=X2)
        self.assertTrue(np.all(X2 == X1))        
        #No explicit alphabet
        self.assertTrue(discrete.divergence_kullbackleibler_symmetrised((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=None, Alphabet_Y=None) == discrete.divergence_kullbackleibler_symmetrised((1,2,1,2,1,2),(2,2,2,2,1,1), Alphabet_X=None, Alphabet_Y=None))
        self.assertTrue(discrete.divergence_kullbackleibler_symmetrised((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=None, Alphabet_Y=None) == discrete.divergence_kullbackleibler_symmetrised((1,2,1,2,1,2),(2,2,2,2,1,1), Alphabet_X=None, Alphabet_Y=None))
        self.assertTrue(discrete.divergence_kullbackleibler_symmetrised((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=None, Alphabet_Y=None) == discrete.divergence_kullbackleibler_symmetrised((1,2,1,2,1,2),(2,2,2,2,1,1), Alphabet_X=None, Alphabet_Y=None))
        #No explicit alphabet -- Cartesian product
        A = discrete.divergence_kullbackleibler_symmetrised(np.array(((1,2,1,2),(2,2,2,1))), estimator=1, Alphabet_X=None, base=3)
        B = discrete.divergence_kullbackleibler_symmetrised(np.array(((1,2,1,2,1,2),(2,2,2,1,1,2))), estimator=0, Alphabet_X=None, base=3)        
        self.assertTrue(np.all(A==B))
        #Larger alphabet
        self.assertTrue(discrete.divergence_kullbackleibler_symmetrised((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=(1,2,3), Alphabet_Y=None) == discrete.divergence_kullbackleibler_symmetrised((1,2,1,2,1,2,3),(2,2,2,2,1,1), Alphabet_X=None, Alphabet_Y=None))
        self.assertTrue(discrete.divergence_kullbackleibler_symmetrised((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=None, Alphabet_Y=(1,2,3)) == discrete.divergence_kullbackleibler_symmetrised((1,2,1,2,1,2),(2,2,2,2,1,1,3), Alphabet_X=None, Alphabet_Y=None))
        self.assertTrue(discrete.divergence_kullbackleibler_symmetrised((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=(1,2,3), Alphabet_Y=(1,2,3)) == discrete.divergence_kullbackleibler_symmetrised((1,2,1,2,1,2,3),(2,2,2,2,1,1,3), Alphabet_X=None, Alphabet_Y=None))        
        #Larger alphabet -- Cartesian product
        A = discrete.divergence_kullbackleibler_symmetrised(np.array(((1,2),(2,2))), estimator=1, Alphabet_X=np.array(((1,2,3),(1,2,3))), base=3)
        B = discrete.divergence_kullbackleibler_symmetrised(np.array(((1,2,1,2,3),(2,2,2,3,1))), estimator=0, Alphabet_X=None, base=3)
        self.assertTrue(np.allclose(A,B))
        #Empty alphabet
        with self.assertRaises(ValueError):
            discrete.divergence_kullbackleibler_symmetrised([1, 2], [1, 2], estimator=1, Alphabet_X=np.array(()), Alphabet_Y=None, base=3)
        with self.assertRaises(ValueError):
            discrete.divergence_kullbackleibler_symmetrised([1, 2], [1, 2], estimator=1, Alphabet_X=None, Alphabet_Y=np.array(()), base=3)
        with self.assertRaises(ValueError):
            discrete.divergence_kullbackleibler_symmetrised([1, 2], estimator=1, Alphabet_X=np.array(()), base=3)            
        #Smaller alphabet
        with self.assertRaises(ValueError):
            discrete.divergence_kullbackleibler_symmetrised([1,2], [1,2], estimator=1, Alphabet_X=[1,2], Alphabet_Y=[1,1], base=3)
        with self.assertRaises(ValueError):
            discrete.divergence_kullbackleibler_symmetrised([1,2], [1,2], estimator=1, Alphabet_X=[1,1], Alphabet_Y=[1,2], base=3)        
        with self.assertRaises(ValueError):
            discrete.divergence_kullbackleibler_symmetrised(np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array(((1,2),(1,None))), base=3)
        #Alphabet with incorrect dimensions
        with self.assertRaises(ValueError):
            discrete.divergence_kullbackleibler_symmetrised(np.array(((1,2),(1,2))), np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array((1,2)), Alphabet_Y=np.array(((1,2),(1,2))), base=3)
        with self.assertRaises(ValueError):
            discrete.divergence_kullbackleibler_symmetrised(np.array(((1,2),(1,2))), np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array(((1,2),(1,2))), Alphabet_Y=np.array((1,2)), base=3)            
        with self.assertRaises(ValueError):
            discrete.divergence_kullbackleibler_symmetrised(np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array((1,2)), base=3)
        
    def test_entropy_conditional(self):
        #Immutability test
        X1 = np.random.randint(16, size=(10,10))
        X1_copy = np.copy(X1)
        X2 = np.random.randint(16, size=(10,10))
        X2_copy = np.copy(X2)
        discrete.entropy_conditional(X1,X2)
        self.assertTrue(np.all(X1_copy == X1) and np.all(X2_copy == X2))
        
        self.assertTrue(discrete.entropy_conditional(np.array((1.0, 2.0, 3.0, 4.0)), np.array((0.5, 0.5, 1.0, 1.0)), base=2) == 1)
        self.assertTrue(discrete.entropy_conditional(np.array((1.0, 2.0, 3.0, 4.0)), np.array((1.0, 2.0, 3.0, 4.0)), base=2) == 0)
        self.assertTrue(discrete.entropy_conditional(np.array((1.0, 2.0, 1.0, 2.0)), np.array((2.0, 2.0, 2.0, 2.0)), base=2) == 1)
        self.assertTrue(abs(discrete.entropy_conditional(np.array((1.0, 2.0, 3.0, 4.0)), np.array((0.5, 0.5, 1.0, 1.0)), base=np.exp(1))-0.693) < 1E-03)
        self.assertTrue(abs(discrete.entropy_conditional(np.array((1.0, 2.0)), np.array((1.0, 1.0)), base=np.exp(1))-0.693) < 1E-03)       
        #Tests using cartesian_product=True                
        self.assertTrue(discrete.entropy_conditional(np.array((1.0, 2.0, 3.0, 4.0)), np.array((0.5, 0.5, 1.0, 1.0)), cartesian_product=True, base=2) == 1)
        self.assertTrue(discrete.entropy_conditional(np.array((1.0, 2.0, 3.0, 4.0)), np.array((1.0, 2.0, 3.0, 4.0)), cartesian_product=True, base=2) == 0)
        self.assertTrue(discrete.entropy_conditional(np.array((1.0, 2.0, 1.0, 2.0)), np.array((2.0, 2.0, 2.0, 2.0)), cartesian_product=True, base=2) == 1)        
        self.assertTrue(abs(discrete.entropy_conditional(np.array((1.0, 2.0, 3.0, 4.0)), np.array((0.5, 0.5, 1.0, 1.0)), cartesian_product=True, base=np.exp(1))-0.693) < 1E-03)
        self.assertTrue(abs(discrete.entropy_conditional(np.array((1.0, 2.0)), np.array((1.0, 1.0)), cartesian_product=True, base=np.exp(1))-0.693) < 1E-03)               
        
        self.assertTrue(discrete.entropy_conditional(np.ones(5),np.ones(5))==0)
        self.assertTrue(discrete.entropy_conditional(np.ones((5,1)).T,np.ones((5,1)).T)==0)
        self.assertTrue(discrete.entropy_conditional(np.ones((5,1)),np.ones((5,1))).size==5)
        self.assertTrue(np.all(discrete.entropy_conditional(np.ones((5,1)),np.ones((5,1)))==0))
        #Tests using cartesian_product=True        
        self.assertTrue(discrete.entropy_conditional(np.ones(5),np.ones(5), cartesian_product=True)==0)
        self.assertTrue(discrete.entropy_conditional(np.ones((5,1)).T,np.ones((5,1)).T, cartesian_product=True)==0)
        self.assertTrue(discrete.entropy_conditional(np.ones((5,1)),np.ones((5,1)), cartesian_product=True).size==25)
        self.assertTrue(np.all(discrete.entropy_conditional(np.ones((5,1)),np.ones((5,1)), cartesian_product=True)==0))        
        
        self.assertTrue(discrete.entropy_conditional(np.array(4),np.array(4)) == 0)
        self.assertTrue(discrete.entropy_conditional(np.array((4,)),np.array((4,))) == 0)
        self.assertTrue(discrete.entropy_conditional(np.ones((1,)),np.ones((1,))) == 0)
        self.assertTrue(discrete.entropy_conditional(np.ones((1,1)),np.ones((1,1))) == 0)        
        #Tests using cartesian_product=True
        self.assertTrue(discrete.entropy_conditional(np.array(4),np.array(4), cartesian_product=True) == 0)
        self.assertTrue(discrete.entropy_conditional(np.array((4,)),np.array((4,)), cartesian_product=True) == 0)
        self.assertTrue(discrete.entropy_conditional(np.ones((1,)),np.ones((1,)), cartesian_product=True) == 0)
        self.assertTrue(discrete.entropy_conditional(np.ones((1,1)),np.ones((1,1)), cartesian_product=True) == 0)        
        
        self.assertTrue(np.all(discrete.entropy_conditional(np.array(((1.0, 2.0, 3.0, 4.0), (1.0, 2.0, 1.0, 2.0))))== np.array(((0, 1),(0, 0)))))
        self.assertTrue(np.all(discrete.entropy_conditional(np.array(((1.0, 2.0, 3.0, 4.0), (1.0, 2.0, 1.0, 2.0))).T,np.array(((1.0, 2.0, 3.0, 4.0), (1.0, 2.0, 1.0, 2.0))).T)==0))
        #Tests using cartesian_product=True        
        self.assertTrue(np.all(discrete.entropy_conditional(np.array(((1.0, 2.0, 3.0, 4.0), (1.0, 2.0, 1.0, 2.0))), cartesian_product=True)== np.array(((0, 1),(0, 0)))))
        H = discrete.entropy_conditional(np.array(((1.0, 2.0, 3.0, 4.0), (1.0, 2.0, 1.0, 2.0))).T,np.array(((1.0, 2.0, 3.0, 4.0), (1.0, 2.0, 1.0, 2.0))).T, cartesian_product=True)
        self.assertTrue(np.all(H[2:4,0:2]) and not np.any(H[0:2,:]) and not np.any(H[:,2:4]))
        
        self.assertTrue(discrete.entropy_conditional(np.array(1),np.array(1)).shape == tuple())
        self.assertTrue(discrete.entropy_conditional(np.array((1,)),np.array((1,))).shape == tuple())
        self.assertTrue(discrete.entropy_conditional(np.ones((1,)),np.ones((1,))).shape == tuple())
        self.assertTrue(discrete.entropy_conditional(np.ones((1,1)),np.ones((1,1))).shape == (1,))
        #Tests using cartesian_product=True
        self.assertTrue(discrete.entropy_conditional(np.array(1),np.array(1),True).shape == tuple())
        self.assertTrue(discrete.entropy_conditional(np.array((1,)),np.array((1,)),True).shape == tuple())
        self.assertTrue(discrete.entropy_conditional(np.ones((1,)),np.ones((1,)),True).shape == tuple())
        self.assertTrue(discrete.entropy_conditional(np.ones((1,1)),np.ones((1,1)),True).shape == (1,1))
        self.assertTrue(discrete.entropy_conditional(np.ones((1,)),np.ones((1,1)),True).shape == (1,))
        self.assertTrue(discrete.entropy_conditional(np.ones((1,1)),np.ones((1,)),True).shape == (1,))
        self.assertTrue(discrete.entropy_conditional(np.ones((1,1)),np.ones((1,1)),True).shape == (1,1))        
        
        X = np.arange(3*4*5).reshape((3,4,5))
        self.assertTrue(isinstance(discrete.entropy_conditional(X,X), np.ndarray))
        self.assertTrue(discrete.entropy_conditional(X,X).shape == (3,4))
        self.assertTrue(np.all(discrete.entropy_conditional(X,X)==0))
        #Tests using cartesian_product=True  
        X = np.arange(3*4*5).reshape((3,4,5))
        self.assertTrue(isinstance(discrete.entropy_conditional(X,X, cartesian_product=True), np.ndarray))
        self.assertTrue(discrete.entropy_conditional(X,X, cartesian_product=True).shape == (3,4,3,4))
        H = discrete.entropy_conditional(X,X, cartesian_product=True)        
        self.assertTrue(np.all(H == 0))
        
        np.random.seed(4759)
        X1 = np.random.rand(10**6)
        X2 = X1 * 0.5;
        Bins1 = np.linspace(0,1,10000)
        P1 = np.digitize(X1, Bins1)
        P2 = np.digitize(X2, Bins1)
        H = discrete.entropy_conditional(P1, P2, base=2)
        self.assertTrue(np.abs(H - 1.0) < 1E-2)        
        
        Scales = np.linspace(0.1,1.0,8).reshape(2,2,2)
        P2 = np.empty((2,2,2,10**6))
        H_predicted = np.empty((2,2,2))
        for i in np.arange(Scales.shape[0]):
            for j in np.arange(Scales.shape[1]):
                for k in np.arange(Scales.shape[2]):
                    X2 = X1 * Scales[i,j,k]
                    P = np.digitize(X2, Bins1)
                    P2[i,j,k,:] = P
                    H_predicted[i,j,k] = discrete.entropy_joint(np.vstack((P1, P)), base=np.exp(1)) - discrete.entropy(P1, base=np.exp(1))
        H = discrete.entropy_conditional(P2, P1, True, base=np.exp(1))
        self.assertTrue(np.all(np.abs(H_predicted - H) == 0))
        
        with self.assertRaises(ValueError):
            discrete.entropy_conditional(np.array(()), np.array((1,2,3)))
        with self.assertRaises(ValueError):
            discrete.entropy_conditional(np.array((1,2,3)), np.array(()))            
        with self.assertRaises(ValueError):
            discrete.entropy_conditional(np.array((np.nan, 1, 2)), np.array((1,2,3)))
        with self.assertRaises(ValueError):
            discrete.entropy_conditional(np.array((1,2,3)), np.array((1,2,np.nan)))
        with self.assertRaises(ValueError):
            discrete.entropy_conditional(np.array((1,2,3)), np.array((1,2,3)), Alphabet_X=(1,2,3,np.nan), Alphabet_Y=(1,2,3))
        with self.assertRaises(ValueError):
            discrete.entropy_conditional(np.array((1,2,3)), np.array((1,2,3)), Alphabet_X=(1,2,3), Alphabet_Y=(1,2,3,np.nan))
        with self.assertRaises(ValueError):
            discrete.entropy_conditional(np.array((1,2,3)), np.array((1,2,3)), fill_value=np.nan)            
        with self.assertRaises(ValueError):            
            discrete.entropy_conditional(np.array((1,2,3)), np.array((4,1,2,3)))
        with self.assertRaises(ValueError):
            discrete.entropy_conditional(np.array(((1,2,3),(4,5,6))), np.array((4,1,2,3)))        
        try:
            discrete.entropy_conditional(np.array(((1,2,3),(4,5,6))), np.array((4,1,2)), cartesian_product=True)
        except ValueError:
            self.fail("raised ValueError unexpectedly")
        with self.assertRaises(ValueError):
            discrete.entropy_conditional(np.array((2, 1)), np.array((2, 1)), base=-1)
            
        #
        # Tests using missing data (basic)
        #
        #Immutability test        
        X1 = np.random.randint(16, size=(10,10)) * 1.0
        X1[0,0] = -1
        X1_copy = np.copy(X1)
        X2 = np.random.randint(16, size=(10,10))
        X2[0,1] = -1
        X2_copy = np.copy(X2)
        discrete.entropy_conditional(X1,X2, fill_value=-1)
        self.assertTrue(np.all(X1_copy == X1) and np.all(X2_copy == X2))
        self.assertFalse(np.all(discrete.entropy_conditional(X1,X2, fill_value=-1) == discrete.entropy_conditional(X1,X2, fill_value=None)))
        
        # Tests using missing data involving masked arrays
        X1 = np.random.randint(16, size=(10,10)) * 1.0
        X1[0,0] = -1
        Mask1 = np.zeros_like(X1)
        Mask1[0,0] = 1
        X2 = np.random.randint(16, size=(10,10)) * 1.0
        X2[0,1] = -1
        Mask2 = np.zeros_like(X2)
        Mask2[0,1] = 1
        self.assertTrue(np.all(discrete.entropy_conditional(X1, X2, fill_value=-1) == discrete.entropy_conditional(np.ma.array(X1,mask=Mask1), X2, fill_value=-1)))
        self.assertTrue(np.all(discrete.entropy_conditional(X1, X2, fill_value=-1) == discrete.entropy_conditional(X1, np.ma.array(X2,mask=Mask2), fill_value=-1)))
        self.assertTrue(np.all(discrete.entropy_conditional(X1, X2, fill_value=-1) == discrete.entropy_conditional(np.ma.array(X1,mask=Mask1), np.ma.array(X2,mask=Mask2), fill_value=-1)))        
        
        ### Tests involving Alphabet and non-ML estimation
        ###
        ###
        #Alphabet immutability 
        X1 = np.random.randint(16, size=(10,10))
        X2 = np.copy(X1)
        discrete.entropy_conditional(X2, X2, estimator=1, Alphabet_X=X2, Alphabet_Y=X2)        
        self.assertTrue(np.all(X2 == X1))
        discrete.entropy_conditional(X2, estimator=1, Alphabet_X=X2)
        self.assertTrue(np.all(X2 == X1))        
        #No explicit alphabet
        self.assertTrue(discrete.entropy_conditional((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=None, Alphabet_Y=None) == discrete.entropy_joint(np.array(((1,1,1,1,2,2,2,2),(1,2,2,2,1,1,2,2))))-discrete.entropy((2,2,2,2,1,1)))
        #No explicit alphabet -- Cartesian product
        self.assertTrue(np.all(discrete.entropy_conditional(np.array(((1,2,1,2),(2,2,2,1))), estimator=1, Alphabet_X=None, base=3) != discrete.entropy_conditional(np.array(((1,2,1,2),(2,2,2,1))), Alphabet_X=None, base=3)))
        #Larger alphabet
        self.assertTrue(discrete.entropy_conditional((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=(1,2,3), Alphabet_Y=None) == discrete.entropy_joint(np.array(((1,1,1,1,2,2,2,2,3,3),(1,2,2,2,1,1,2,2,1,2)))) - discrete.entropy((1,1,2,2,2,2)))
        self.assertTrue(np.allclose(discrete.entropy_conditional((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=None, Alphabet_Y=(1,2,3)), discrete.entropy_joint(np.array(((1,1,1,1,1,2,2,2,2,2,),(1,2,2,2,3,1,1,2,2,3)))) - discrete.entropy((1,1,2,2,2,2,3))))
        self.assertTrue(np.allclose(discrete.entropy_conditional((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=(1,2,3), Alphabet_Y=(1,2,3)), discrete.entropy_joint(np.array(((1,1,1,1,1,2,2,2,2,2,3,3,3,),(1,2,2,2,3,1,1,2,2,3,1,2,3,)))) - discrete.entropy((1,1,2,2,2,2,3))))        
        #Larger alphabet -- Cartesian product
        self.assertTrue(np.all(discrete.entropy_conditional(np.array(((1,2),(2,2))), estimator=1, Alphabet_X=np.array(((1,2,3),(1,2,3))), base=3) != discrete.entropy_conditional(np.array(((1,2),(2,2))), Alphabet_X=np.array(((1,2,3),(1,2,3))), base=3)))
        #Empty alphabet
        with self.assertRaises(ValueError):
            discrete.entropy_conditional([1, 2], [1, 2], estimator=1, Alphabet_X=np.array(()), Alphabet_Y=None, base=3)
        with self.assertRaises(ValueError):
            discrete.entropy_conditional([1, 2], [1, 2], estimator=1, Alphabet_X=None, Alphabet_Y=np.array(()), base=3)
        with self.assertRaises(ValueError):
            discrete.entropy_conditional([1, 2], estimator=1, Alphabet_X=np.array(()), base=3)            
        #Smaller alphabet
        with self.assertRaises(ValueError):
            discrete.entropy_conditional([1,2], [1,2], estimator=1, Alphabet_X=[1,2], Alphabet_Y=[1,1], base=3)
        with self.assertRaises(ValueError):
            discrete.entropy_conditional([1,2], [1,2], estimator=1, Alphabet_X=[1,1], Alphabet_Y=[1,2], base=3)        
        with self.assertRaises(ValueError):
            discrete.entropy_conditional(np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array(((1,2),(1,None))), base=3)
        #Alphabet with incorrect dimensions
        with self.assertRaises(ValueError):
            discrete.entropy_conditional(np.array(((1,2),(1,2))), np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array((1,2)), Alphabet_Y=np.array(((1,2),(1,2))), base=3)
        with self.assertRaises(ValueError):
            discrete.entropy_conditional(np.array(((1,2),(1,2))), np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array(((1,2),(1,2))), Alphabet_Y=np.array((1,2)), base=3)            
        with self.assertRaises(ValueError):
            discrete.entropy_conditional(np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array((1,2)), base=3)        
                
    def test_information_mutual(self):
        #Immutability test
        X1 = np.random.randint(16, size=(10,10))
        X1_copy = np.copy(X1)
        X2 = np.random.randint(16, size=(10,10))
        X2_copy = np.copy(X2)
        discrete.information_mutual(X1,X2)
        self.assertTrue(np.all(X1_copy == X1) and np.all(X2_copy == X2))
        
        self.assertTrue(discrete.information_mutual(np.array((1.0, 2.0, 3.0, 4.0)), np.array((0.5, 0.5, 1.0, 1.0)), base=2) == 1)
        self.assertTrue(discrete.information_mutual(np.array((1.0, 2.0, 3.0, 4.0)), np.array((1.0, 2.0, 3.0, 4.0)), base=2) == 2)
        self.assertTrue(discrete.information_mutual(np.array((1.0, 2.0, 1.0, 2.0)), np.array((2.0, 2.0, 2.0, 2.0)), base=2) == 0)
        self.assertTrue(abs(discrete.information_mutual(np.array((1.0, 2.0, 3.0, 4.0)), np.array((0.5, 0.5, 1.0, 1.0)), base=np.exp(1))-0.693) < 1E-03)
        self.assertTrue(abs(discrete.information_mutual(np.array((1.0, 2.0)), np.array((2.0, 1.0)), base=np.exp(1))-0.693) < 1E-03)       
        #Tests using cartesian_product=True
        self.assertTrue(discrete.information_mutual(np.array((1.0, 2.0, 3.0, 4.0)), np.array((0.5, 0.5, 1.0, 1.0)), cartesian_product=True, base=2) == 1)
        self.assertTrue(discrete.information_mutual(np.array((1.0, 2.0, 3.0, 4.0)), np.array((1.0, 2.0, 3.0, 4.0)), cartesian_product=True, base=2) == 2)
        self.assertTrue(discrete.information_mutual(np.array((1.0, 2.0, 1.0, 2.0)), np.array((2.0, 2.0, 2.0, 2.0)), cartesian_product=True, base=2) == 0)
        self.assertTrue(abs(discrete.information_mutual(np.array((1.0, 2.0, 3.0, 4.0)), np.array((0.5, 0.5, 1.0, 1.0)), cartesian_product=True, base=np.exp(1))-0.693) < 1E-03)
        self.assertTrue(abs(discrete.information_mutual(np.array((1.0, 2.0)), np.array((2.0, 1.0)), cartesian_product=True, base=np.exp(1))-0.693) < 1E-03)        
        
        self.assertTrue(discrete.information_mutual(np.ones(5),np.ones(5))==0)
        self.assertTrue(discrete.information_mutual(np.ones((5,1)).T,np.ones((5,1)).T)==0)
        self.assertTrue(discrete.information_mutual(np.ones((5,1)),np.ones((5,1))).size==5)
        self.assertTrue(np.all(discrete.information_mutual(np.ones((5,1)),np.ones((5,1)))==0))
        #Tests using cartesian_product=True        
        self.assertTrue(discrete.information_mutual(np.ones(5),np.ones(5), cartesian_product=True)==0)
        self.assertTrue(discrete.information_mutual(np.ones((5,1)).T,np.ones((5,1)).T, cartesian_product=True)==0)
        self.assertTrue(discrete.information_mutual(np.ones((5,1)),np.ones((5,1)), cartesian_product=True).size==25)
        self.assertTrue(np.all(discrete.information_mutual(np.ones((5,1)),np.ones((5,1)), cartesian_product=True)==0))        
        
        self.assertTrue(discrete.information_mutual(np.array(4),np.array(4)) == 0)
        self.assertTrue(discrete.information_mutual(np.array((4,)),np.array((4,))) == 0)
        self.assertTrue(discrete.information_mutual(np.ones((1,)),np.ones((1,))) == 0)
        self.assertTrue(discrete.information_mutual(np.ones((1,1)),np.ones((1,1))) == 0)        
        #Tests using cartesian_product=True
        self.assertTrue(discrete.information_mutual(np.array(4),np.array(4), cartesian_product=True) == 0)
        self.assertTrue(discrete.information_mutual(np.array((4,)),np.array((4,)), cartesian_product=True) == 0)
        self.assertTrue(discrete.information_mutual(np.ones((1,)),np.ones((1,)), cartesian_product=True) == 0)
        self.assertTrue(discrete.information_mutual(np.ones((1,1)),np.ones((1,1)), cartesian_product=True) == 0)
        
        self.assertTrue(discrete.information_mutual(np.array(1),np.array(1)).shape == tuple())
        self.assertTrue(discrete.information_mutual(np.array((1,)),np.array((1,))).shape == tuple())
        self.assertTrue(discrete.information_mutual(np.ones((1,)),np.ones((1,))).shape == tuple())
        self.assertTrue(discrete.information_mutual(np.ones((1,1)),np.ones((1,1))).shape == (1,))
        #Tests using cartesian_product=True
        self.assertTrue(discrete.information_mutual(np.array(1),np.array(1),True).shape == tuple())
        self.assertTrue(discrete.information_mutual(np.array((1,)),np.array((1,)),True).shape == tuple())
        self.assertTrue(discrete.information_mutual(np.ones((1,)),np.ones((1,)),True).shape == tuple())
        self.assertTrue(discrete.information_mutual(np.ones((1,1)),np.ones((1,1)),True).shape == (1,1))
        self.assertTrue(discrete.information_mutual(np.ones((1,)),np.ones((1,1)),True).shape == (1,))
        self.assertTrue(discrete.information_mutual(np.ones((1,1)),np.ones((1,)),True).shape == (1,))
        self.assertTrue(discrete.information_mutual(np.ones((1,1)),np.ones((1,1)),True).shape == (1,1))        
        
        self.assertTrue(np.all(discrete.information_mutual(np.array(((1.0, 2.0, 3.0, 4.0), (1.0, 2.0, 1.0, 2.0))))== np.array(((2, 1),(1, 1)))))
        self.assertTrue(np.all(discrete.information_mutual(np.array(((1.0, 2.0, 3.0, 4.0), (1.0, 2.0, 1.0, 2.0))).T,np.array(((1.0, 2.0, 3.0, 4.0), (1.0, 2.0, 1.0, 2.0))).T)==np.array((0,0,1,1))))
        #Tests using cartesian_product=True        
        self.assertTrue(np.all(discrete.information_mutual(np.array(((1.0, 2.0, 3.0, 4.0), (1.0, 2.0, 1.0, 2.0))), cartesian_product=True)== np.array(((2, 1),(1, 1)))))
        H = discrete.information_mutual(np.array(((1.0, 2.0, 3.0, 4.0), (1.0, 2.0, 1.0, 2.0))).T,np.array(((1.0, 2.0, 3.0, 4.0), (1.0, 2.0, 1.0, 2.0))).T, cartesian_product=True)
        self.assertTrue(np.all(H[2:4,2:4]) and not np.any(H[0:2,:]) and not np.any(H[:,0:2]))
        
        X = np.arange(3*4*8).reshape((3,4,8))
        self.assertTrue(isinstance(discrete.information_mutual(X,X), np.ndarray))
        self.assertTrue(discrete.information_mutual(X,X).shape == (3,4))
        self.assertTrue(np.all(discrete.information_mutual(X,X)==3))
        #Tests using cartesian_product=True  
        X = np.arange(3*4*8).reshape((3,4,8))
        self.assertTrue(isinstance(discrete.information_mutual(X,X, cartesian_product=True), np.ndarray))
        self.assertTrue(discrete.information_mutual(X,X, cartesian_product=True).shape == (3,4,3,4))
        H = discrete.information_mutual(X,X, cartesian_product=True)        
        self.assertTrue(np.all(H == 3))
                
        #Distribution test based on entropy/joint entropy/conditional entropy
        Bins1 = np.linspace(-5,5,10000)
        P1 = norm.pdf(Bins1,loc=-0.5)
        P2 = norm.pdf(Bins1,loc=-0.0)
        P3 = norm.pdf(Bins1,loc=+0.5)     
        P1 = P1 / np.sum(P1)
        P2 = P2 / np.sum(P2)
        P3 = P3 / np.sum(P3)        
        #Sample from component distributions
        I = np.random.choice(P1.size, 10**6, p=P1)
        J = np.random.choice(P1.size, 10**6, p=P2)
        K = np.random.choice(P1.size, 10**6, p=P3)
        #Create correlated random variables
        X = I + J
        Y = J + K
        
        I_observed = discrete.information_mutual(X, Y, base=3)
        self.assertTrue(np.allclose(I_observed, discrete.entropy(X, base=3) - discrete.entropy_conditional(X, Y, base=3)))
        self.assertTrue(np.allclose(I_observed, discrete.entropy(Y, base=3) - discrete.entropy_conditional(Y, X, base=3)))
        self.assertTrue(np.allclose(I_observed, discrete.entropy(X, base=3) + discrete.entropy(Y, base=3) - discrete.entropy_joint(np.vstack((Y, X)), base=3)))
        
        #
        # Tests using missing data (basic)
        #
        #Immutability test        
        X1 = np.random.randint(16, size=(10,10)) * 1.0
        X1[0,0] = -1
        X1_copy = np.copy(X1)
        X2 = np.random.randint(16, size=(10,10))
        X2[0,1] = -1
        X2_copy = np.copy(X2)
        discrete.information_mutual(X1,X2, fill_value=-1)
        self.assertTrue(np.all(X1_copy == X1) and np.all(X2_copy == X2))
        self.assertFalse(np.all(discrete.information_mutual(X1,X2, fill_value=-1) == discrete.information_mutual(X1,X2, fill_value=None)))
        
        ### Tests involving Alphabet and non-ML estimation
        ###
        ###
        #Alphabet immutability 
        X1 = np.random.randint(16, size=(10,10))
        X2 = np.copy(X1)
        discrete.information_mutual(X2, X2, estimator=1, Alphabet_X=X2, Alphabet_Y=X2)        
        self.assertTrue(np.all(X2 == X1))
        discrete.information_mutual(X2, estimator=1, Alphabet_X=X2)
        self.assertTrue(np.all(X2 == X1))        
        #No explicit alphabet
        self.assertTrue(discrete.information_mutual((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=None, Alphabet_Y=None, base=3) == discrete.entropy((1,2,1,2), estimator=1, base=3)-discrete.entropy_conditional((1,2,1,2),(2,2,2,1), estimator=1, base=3))
        #No explicit alphabet -- Cartesian product
        self.assertTrue(np.all(discrete.information_mutual(np.array(((1,2,1,2),(2,2,2,1))), estimator=1, Alphabet_X=None, base=3) != discrete.information_mutual(np.array(((1,2,1,2),(2,2,2,1))), Alphabet_X=None, base=3)))
        #Larger alphabet
        self.assertTrue(discrete.information_mutual((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=(1,2,3), Alphabet_Y=None) == discrete.entropy((1,2,1,2), estimator=1, Alphabet_X=(1,2,3)) - discrete.entropy_conditional((1,2,1,2),(2,2,2,1),estimator=1, Alphabet_X=(1,2,3), Alphabet_Y=None))
        self.assertTrue(discrete.information_mutual((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=None, Alphabet_Y=(1,2,3)) == discrete.entropy((1,2,1,2), estimator=1, Alphabet_X=None) - discrete.entropy_conditional((1,2,1,2),(2,2,2,1),estimator=1, Alphabet_X=None, Alphabet_Y=(1,2,3)))
        self.assertTrue(discrete.information_mutual((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=(1,2,3), Alphabet_Y=(1,2,3)) == discrete.entropy((1,2,1,2), estimator=1, Alphabet_X=(1,2,3)) - discrete.entropy_conditional((1,2,1,2),(2,2,2,1),estimator=1, Alphabet_X=(1,2,3), Alphabet_Y=(1,2,3)))        
        #Larger alphabet -- Cartesian product
        self.assertTrue(np.all(discrete.information_mutual(np.array(((1,2,1,2),(2,2,2,1))), estimator=1, Alphabet_X=np.array(((1,2,3),(1,2,3))), base=3) != discrete.information_mutual(np.array(((2,2,2,1),(2,2,2,1))), Alphabet_X=np.array(((1,2,3),(1,2,3))), base=3)))        
        #Empty alphabet
        with self.assertRaises(ValueError):
            discrete.information_mutual([1, 2], [1, 2], estimator=1, Alphabet_X=np.array(()), Alphabet_Y=None, base=3)
        with self.assertRaises(ValueError):
            discrete.information_mutual([1, 2], [1, 2], estimator=1, Alphabet_X=None, Alphabet_Y=np.array(()), base=3)
        with self.assertRaises(ValueError):
            discrete.information_mutual([1, 2], estimator=1, Alphabet_X=np.array(()), base=3)            
        #Smaller alphabet
        with self.assertRaises(ValueError):
            discrete.information_mutual([1,2], [1,2], estimator=1, Alphabet_X=[1,2], Alphabet_Y=[1,1], base=3)
        with self.assertRaises(ValueError):
            discrete.information_mutual([1,2], [1,2], estimator=1, Alphabet_X=[1,1], Alphabet_Y=[1,2], base=3)        
        with self.assertRaises(ValueError):
            discrete.information_mutual(np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array(((1,2),(1,None))), base=3)
        #Alphabet with incorrect dimensions
        with self.assertRaises(ValueError):
            discrete.information_mutual(np.array(((1,2),(1,2))), np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array((1,2)), Alphabet_Y=np.array(((1,2),(1,2))), base=3)
        with self.assertRaises(ValueError):
            discrete.information_mutual(np.array(((1,2),(1,2))), np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array(((1,2),(1,2))), Alphabet_Y=np.array((1,2)), base=3)            
        with self.assertRaises(ValueError):
            discrete.information_mutual(np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array((1,2)), base=3)        
        
    def test_information_variation(self):
        #Immutability test
        X1 = np.random.randint(16, size=(10,10))
        X1_copy = np.copy(X1)
        X2 = np.random.randint(16, size=(10,10))
        X2_copy = np.copy(X2)
        discrete.information_variation(X1,X2)
        self.assertTrue(np.all(X1_copy == X1) and np.all(X2_copy == X2))        

        self.assertTrue(np.allclose(discrete.information_variation(np.array((1,2,1)), np.array((1,2,2))), 4.0/3))
        self.assertTrue(np.allclose(discrete.information_variation(np.array((1,2,1)), np.array((1,2,2))), 2 * discrete.entropy_conditional(np.array((1,2,1)), np.array((1,2,2)))))
        
        self.assertTrue(discrete.information_variation(np.array(1),np.array(1)).shape == tuple())
        self.assertTrue(discrete.information_variation(np.array((1,)),np.array((1,))).shape == tuple())
        self.assertTrue(discrete.information_variation(np.ones((1,)),np.ones((1,))).shape == tuple())
        self.assertTrue(discrete.information_variation(np.ones((1,1)),np.ones((1,1))).shape == (1,))
        #Tests using cartesian_product=True
        self.assertTrue(discrete.information_variation(np.array(1),np.array(1),True).shape == tuple())
        self.assertTrue(discrete.information_variation(np.array((1,)),np.array((1,)),True).shape == tuple())
        self.assertTrue(discrete.information_variation(np.ones((1,)),np.ones((1,)),True).shape == tuple())
        self.assertTrue(discrete.information_variation(np.ones((1,1)),np.ones((1,1)),True).shape == (1,1))
        self.assertTrue(discrete.information_variation(np.ones((1,)),np.ones((1,1)),True).shape == (1,))
        self.assertTrue(discrete.information_variation(np.ones((1,1)),np.ones((1,)),True).shape == (1,))
        self.assertTrue(discrete.information_variation(np.ones((1,1)),np.ones((1,1)),True).shape == (1,1))        
                
        X = np.array(((1,2,1),(2,2,1),(1,1,2)))
        Y = np.array(((1,2,1),(1,2,2)))
        H = discrete.information_variation(X,Y, True)
        self.assertTrue(np.allclose(H, np.array(((0.0,4.0/3),(4.0/3,4.0/3),(4.0/3,4.0/3)))))
        H = discrete.information_variation(Y)
        self.assertTrue(np.allclose(H, np.array(((0.0,4.0/3),(4.0/3,0)))))
                
        #
        # Tests using missing data (basic)
        #
        #Immutability test        
        X1 = np.random.randint(16, size=(10,10)) * 1.0
        X1[0,0] = -1
        X1_copy = np.copy(X1)
        X2 = np.random.randint(16, size=(10,10))
        X2[0,1] = -1
        X2_copy = np.copy(X2)
        discrete.information_variation(X1,X2, fill_value=-1)
        self.assertTrue(np.all(X1_copy == X1) and np.all(X2_copy == X2))
        self.assertFalse(np.all(discrete.information_variation(X1,X2, fill_value=-1) == discrete.information_variation(X1,X2, fill_value=None)))
        
        ### Tests involving Alphabet and non-ML estimation
        ###
        ###
        #Alphabet immutability 
        X1 = np.random.randint(16, size=(10,10))
        X2 = np.copy(X1)
        discrete.information_variation(X2, X2, estimator=1, Alphabet_X=X2, Alphabet_Y=X2)        
        self.assertTrue(np.all(X2 == X1))
        discrete.information_variation(X2, estimator=1, Alphabet_X=X2)
        self.assertTrue(np.all(X2 == X1))        
        #No explicit alphabet
        self.assertTrue(discrete.information_variation((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=None, Alphabet_Y=None, base=3) == discrete.entropy_conditional((1,2,1,2),(2,2,2,1), estimator=1, base=3)+discrete.entropy_conditional((2,2,2,1),(1,2,1,2), estimator=1, base=3))
        #No explicit alphabet -- Cartesian product
        A = discrete.information_variation(np.array(((1,2,1,2),(2,2,2,1))), estimator=1, Alphabet_X=None, Alphabet_Y=None, base=3)
        B = discrete.entropy_conditional(np.array(((1,2,1,2),(2,2,2,1))), estimator=1, base=3)+discrete.entropy_conditional(np.array(((2,2,2,1),(1,2,1,2))), estimator=1, base=3)
        self.assertTrue(A[0,1] == B[0,1] and A[1,0] == B[0,1])
        #Larger alphabet
        self.assertTrue(discrete.information_variation((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=(1,2,3), Alphabet_Y=None) == discrete.entropy_conditional((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=(1,2,3), Alphabet_Y=None) + discrete.entropy_conditional((2,2,2,1),(1,2,1,2),estimator=1, Alphabet_X=None, Alphabet_Y=(1,2,3)))
        self.assertTrue(discrete.information_variation((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=None, Alphabet_Y=(1,2,3)) == discrete.entropy_conditional((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=None, Alphabet_Y=(1,2,3)) + discrete.entropy_conditional((2,2,2,1),(1,2,1,2),estimator=1, Alphabet_X=(1,2,3), Alphabet_Y=None))
        self.assertTrue(discrete.information_variation((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=(1,2,3), Alphabet_Y=(1,2,3)) == discrete.entropy_conditional((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=(1,2,3), Alphabet_Y=(1,2,3)) + discrete.entropy_conditional((2,2,2,1),(1,2,1,2),estimator=1, Alphabet_X=(1,2,3), Alphabet_Y=(1,2,3)))        
        #Larger alphabet -- Cartesian product
        self.assertTrue(np.all(discrete.information_variation(np.array(((1,2,1,2),(2,2,2,1))), estimator=1, Alphabet_X=np.array(((1,2,3),(1,2,3))), base=3) != discrete.information_variation(np.array(((2,2,2,1),(2,2,2,1))), Alphabet_X=np.array(((1,2,3),(1,2,3))), base=3)))
        #Empty alphabet
        with self.assertRaises(ValueError):
            discrete.information_variation([1, 2], [1, 2], estimator=1, Alphabet_X=np.array(()), Alphabet_Y=None, base=3)
        with self.assertRaises(ValueError):
            discrete.information_variation([1, 2], [1, 2], estimator=1, Alphabet_X=None, Alphabet_Y=np.array(()), base=3)
        with self.assertRaises(ValueError):
            discrete.information_variation([1, 2], estimator=1, Alphabet_X=np.array(()), base=3)            
        #Smaller alphabet
        with self.assertRaises(ValueError):
            discrete.information_variation([1,2], [1,2], estimator=1, Alphabet_X=[1,2], Alphabet_Y=[1,1], base=3)
        with self.assertRaises(ValueError):
            discrete.information_variation([1,2], [1,2], estimator=1, Alphabet_X=[1,1], Alphabet_Y=[1,2], base=3)        
        with self.assertRaises(ValueError):
            discrete.information_variation(np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array(((1,2),(1,None))), base=3)
        #Alphabet with incorrect dimensions
        with self.assertRaises(ValueError):
            discrete.information_variation(np.array(((1,2),(1,2))), np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array((1,2)), Alphabet_Y=np.array(((1,2),(1,2))), base=3)
        with self.assertRaises(ValueError):
            discrete.information_variation(np.array(((1,2),(1,2))), np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array(((1,2),(1,2))), Alphabet_Y=np.array((1,2)), base=3)            
        with self.assertRaises(ValueError):
            discrete.information_variation(np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array((1,2)), base=3)        
        
    def test_information_mutual_normalised(self):
        #Immutability test
        X1 = np.random.randint(16, size=(10,10))
        X1_copy = np.copy(X1)
        X2 = np.random.randint(16, size=(10,10))
        X2_copy = np.copy(X2)
        discrete.information_mutual_normalised(X1,X2)
        self.assertTrue(np.all(X1_copy == X1) and np.all(X2_copy == X2))
        
        self.assertTrue(discrete.information_mutual_normalised(np.array((1,2,2)), np.array((2,1,2))) == discrete.information_mutual(np.array((1,2,2)), np.array((2,1,2)), base=2)/discrete.entropy(np.array((2,1,2)), base=2))
        self.assertTrue(discrete.information_mutual_normalised(np.array((1,2,2)), np.array((2,1,2)), 'Y') == discrete.information_mutual(np.array((1,2,2)), np.array((2,1,2)), base=2)/discrete.entropy(np.array((2,1,2)), base=2))        
        self.assertTrue(discrete.information_mutual_normalised(np.array((1,2,2)), np.array((2,1,2)), 'X') == discrete.information_mutual(np.array((1,2,2)), np.array((2,1,2)), base=2)/discrete.entropy(np.array((1,2,2)), base=2))        
        self.assertTrue(np.allclose(discrete.information_mutual_normalised(np.array((1,2,2)), np.array((2,1,2)), ' x + Y '), discrete.information_mutual(np.array((1,2,2)), np.array((2,1,2)), base=np.exp(1))/(discrete.entropy(np.array((1,2,2)), base=np.exp(1))+discrete.entropy(np.array((2,1,2)), base=np.exp(1)))))        
        self.assertTrue(np.allclose(discrete.information_mutual_normalised(np.array((1,2,2)), np.array((2,1,2)), 'MIN'), discrete.information_mutual(np.array((1,2,2)), np.array((2,1,2)))/np.min(discrete.entropy(np.array((1,2,2))),discrete.entropy(np.array((2,1,2))))))        
        self.assertTrue(np.allclose(discrete.information_mutual_normalised(np.array((1,2,2)), np.array((2,1,2)), 'MAX'), discrete.information_mutual(np.array((1,2,2)), np.array((2,1,2)))/np.min(discrete.entropy(np.array((1,2,2))),discrete.entropy(np.array((2,1,2))))))        
        self.assertTrue(np.allclose(discrete.information_mutual_normalised(np.array((1,2,2)), np.array((2,1,2)), 'XY'), discrete.information_mutual(np.array((1,2,2)), np.array((2,1,2)))/discrete.entropy_joint(np.vstack((np.array((2,1,2)), np.array((1,2,2)))))))
        self.assertTrue(np.allclose(discrete.information_mutual_normalised(np.array((1,2,2)), np.array((2,1,2)), 'SQRT'), discrete.information_mutual(np.array((1,2,2)), np.array((2,1,2)))/np.sqrt(discrete.entropy(np.array((1,2,2)))*discrete.entropy(np.array((2,1,2))))))        
        #Tests using cartesian_product=True
        self.assertTrue(discrete.information_mutual_normalised(np.array((1,2,2)), np.array((2,1,2)), cartesian_product=True) == discrete.information_mutual(np.array((1,2,2)), np.array((2,1,2)), cartesian_product=True, base=2)/discrete.entropy(np.array((2,1,2)),base=2))
        self.assertTrue(discrete.information_mutual_normalised(np.array((1,2,2)), np.array((2,1,2)), 'Y', cartesian_product=True) == discrete.information_mutual(np.array((1,2,2)), np.array((2,1,2)), cartesian_product=True,base=2)/discrete.entropy(np.array((2,1,2)),base=2))        
        self.assertTrue(discrete.information_mutual_normalised(np.array((1,2,2)), np.array((2,1,2)), 'X', cartesian_product=True) == discrete.information_mutual(np.array((1,2,2)), np.array((2,1,2)), cartesian_product=True,base=2)/discrete.entropy(np.array((1,2,2)),base=2))        
        self.assertTrue(np.allclose(discrete.information_mutual_normalised(np.array((1,2,2)), np.array((2,1,2)), ' x + Y ', cartesian_product=True), discrete.information_mutual(np.array((1,2,2)), np.array((2,1,2)), cartesian_product=True,base=np.exp(1))/(discrete.entropy(np.array((1,2,2)),base=np.exp(1))+discrete.entropy(np.array((2,1,2)),base=np.exp(1)))))
        self.assertTrue(np.allclose(discrete.information_mutual_normalised(np.array((1,2,2)), np.array((2,1,2)), 'MIN', cartesian_product=True), discrete.information_mutual(np.array((1,2,2)), np.array((2,1,2)))/np.min(discrete.entropy(np.array((1,2,2))),discrete.entropy(np.array((2,1,2))))))        
        self.assertTrue(np.allclose(discrete.information_mutual_normalised(np.array((1,2,2)), np.array((2,1,2)), 'MAX', cartesian_product=True), discrete.information_mutual(np.array((1,2,2)), np.array((2,1,2)))/np.min(discrete.entropy(np.array((1,2,2))),discrete.entropy(np.array((2,1,2))))))        
        self.assertTrue(np.allclose(discrete.information_mutual_normalised(np.array((1,2,2)), np.array((2,1,2)), 'XY', cartesian_product=True), discrete.information_mutual(np.array((1,2,2)), np.array((2,1,2)))/discrete.entropy_joint(np.vstack((np.array((2,1,2)), np.array((1,2,2)))))))
        self.assertTrue(np.allclose(discrete.information_mutual_normalised(np.array((1,2,2)), np.array((2,1,2)), 'SQRT', cartesian_product=True), discrete.information_mutual(np.array((1,2,2)), np.array((2,1,2)))/np.sqrt(discrete.entropy(np.array((1,2,2)))*discrete.entropy(np.array((2,1,2))))))        
        
        self.assertTrue(discrete.information_mutual_normalised(np.ones(5),np.ones(5)).size==1)
        self.assertTrue(np.isnan(discrete.information_mutual_normalised(np.ones(5),np.ones(5))))        
        self.assertTrue(discrete.information_mutual_normalised(np.ones((5,1)).T,np.ones((5,1)).T).size==1)
        self.assertTrue(np.isnan(discrete.information_mutual_normalised(np.ones((5,1)).T,np.ones((5,1)).T)))
        self.assertTrue(np.all(np.isnan(discrete.information_mutual_normalised(np.ones((5,1)),np.ones((5,1))))))
        with np.errstate(invalid='ignore', divide='ignore'):       
            for S in ('X', 'Y', 'X+Y', 'MIN', 'MAX', 'XY', 'SQRT'):
                self.assertTrue(discrete.information_mutual_normalised(np.ones(5),np.ones(5), S).size==1)
                self.assertTrue(np.isnan(discrete.information_mutual_normalised(np.ones(5),np.ones(5), S)))        
                self.assertTrue(discrete.information_mutual_normalised(np.ones((5,1)).T,np.ones((5,1)).T, S).size==1)
                self.assertTrue(np.isnan(discrete.information_mutual_normalised(np.ones((5,1)).T,np.ones((5,1)).T, S)))
                self.assertTrue(np.all(np.isnan(discrete.information_mutual_normalised(np.ones((5,1)),np.ones((5,1)), S))))   
        #Tests using cartesian_product=True
        self.assertTrue(discrete.information_mutual_normalised(np.ones(5),np.ones(5), cartesian_product=True).size==1)
        self.assertTrue(np.isnan(discrete.information_mutual_normalised(np.ones(5),np.ones(5), cartesian_product=True)))        
        self.assertTrue(discrete.information_mutual_normalised(np.ones((5,1)).T,np.ones((5,1)).T, cartesian_product=True).size==1)
        self.assertTrue(np.isnan(discrete.information_mutual_normalised(np.ones((5,1)).T,np.ones((5,1)).T, cartesian_product=True)))
        self.assertTrue(np.all(np.isnan(discrete.information_mutual_normalised(np.ones((5,1)),np.ones((5,1)), cartesian_product=True))))
        with np.errstate(invalid='ignore', divide='ignore'):       
            for S in ('X', 'Y', 'X+Y', 'MIN', 'MAX', 'XY', 'SQRT'):
                self.assertTrue(discrete.information_mutual_normalised(np.ones(5),np.ones(5), S, cartesian_product=True).size==1)
                self.assertTrue(np.isnan(discrete.information_mutual_normalised(np.ones(5),np.ones(5), S, cartesian_product=True)))        
                self.assertTrue(discrete.information_mutual_normalised(np.ones((5,1)).T,np.ones((5,1)).T, S, cartesian_product=True).size==1)
                self.assertTrue(np.isnan(discrete.information_mutual_normalised(np.ones((5,1)).T,np.ones((5,1)).T, S, cartesian_product=True)))
                self.assertTrue(np.all(np.isnan(discrete.information_mutual_normalised(np.ones((5,1)),np.ones((5,1)), S, cartesian_product=True))))        
        
        self.assertTrue(np.isnan(discrete.information_mutual_normalised(np.array(4),np.array(4))))
        self.assertTrue(np.isnan(discrete.information_mutual_normalised(np.array((4,)),np.array((4,)))))
        self.assertTrue(np.isnan(discrete.information_mutual_normalised(np.ones((1,)),np.ones((1,)))))
        self.assertTrue(np.isnan(discrete.information_mutual_normalised(np.ones((1,1)),np.ones((1,1)))))
        with np.errstate(invalid='ignore', divide='ignore'):       
            for S in ('X', 'Y', 'X+Y', 'MIN', 'MAX', 'XY', 'SQRT'):
                self.assertTrue(np.isnan(discrete.information_mutual_normalised(np.array(4),np.array(4), S)))
                self.assertTrue(np.isnan(discrete.information_mutual_normalised(np.array((4,)),np.array((4,)), S)))
                self.assertTrue(np.isnan(discrete.information_mutual_normalised(np.ones((1,)),np.ones((1,)), S)))
                self.assertTrue(np.isnan(discrete.information_mutual_normalised(np.ones((1,1)),np.ones((1,1)), S)))             
        #Tests using cartesian_product=True
        self.assertTrue(np.isnan(discrete.information_mutual_normalised(np.array(4),np.array(4), cartesian_product=True)))
        self.assertTrue(np.isnan(discrete.information_mutual_normalised(np.array((4,)),np.array((4,)), cartesian_product=True)))
        self.assertTrue(np.isnan(discrete.information_mutual_normalised(np.ones((1,)),np.ones((1,)), cartesian_product=True)))
        self.assertTrue(np.isnan(discrete.information_mutual_normalised(np.ones((1,1)),np.ones((1,1)), cartesian_product=True)))
        with np.errstate(invalid='ignore', divide='ignore'):       
            for S in ('X', 'Y', 'X+Y', 'MIN', 'MAX', 'XY', 'SQRT'):
                self.assertTrue(np.isnan(discrete.information_mutual_normalised(np.array(4),np.array(4), S, cartesian_product=True)))
                self.assertTrue(np.isnan(discrete.information_mutual_normalised(np.array((4,)),np.array((4,)), S, cartesian_product=True)))
                self.assertTrue(np.isnan(discrete.information_mutual_normalised(np.ones((1,)),np.ones((1,)), S, cartesian_product=True)))
                self.assertTrue(np.isnan(discrete.information_mutual_normalised(np.ones((1,1)),np.ones((1,1)), S, cartesian_product=True)))        

        self.assertTrue(np.all(discrete.information_mutual_normalised(np.array(((1,2,1), (1,2,1))),np.array(((1,2,1), (1,2,1))))==1))
        self.assertTrue(np.all(np.abs(discrete.information_mutual_normalised(np.array(((2,2,1,1), (2,1,1,1))), np.array(((2,2,1,1), (2,1,1,1))), norm_factor='X')-np.array((1,1))) < 0.01))
        self.assertTrue(np.all(np.abs(discrete.information_mutual_normalised(np.array(((2,2,1,1), (2,1,1,1))), np.array(((2,2,1,1), (2,1,1,1))), norm_factor='Y')-np.array((1,1))) < 0.01))        
        self.assertTrue(np.all(np.abs(discrete.information_mutual_normalised(np.array(((2,2,1,1), (2,1,1,1))), np.array(((2,2,1,1), (2,1,1,1))), norm_factor='X+Y')-np.array((0.5,0.5))) < 0.01))
        self.assertTrue(np.all(np.abs(discrete.information_mutual_normalised(np.array(((2,2,1,1), (2,1,1,1))), np.array(((2,2,1,1), (2,1,1,1))), norm_factor='MIN')-np.array((1,1))) < 0.01))
        self.assertTrue(np.all(np.abs(discrete.information_mutual_normalised(np.array(((2,2,1,1), (2,1,1,1))), np.array(((2,2,1,1), (2,1,1,1))), norm_factor='MAX')-np.array((1,1))) < 0.01))        
        self.assertTrue(np.all(np.abs(discrete.information_mutual_normalised(np.array(((2,2,1,1), (2,1,1,1))), np.array(((2,2,1,1), (2,1,1,1))), norm_factor='SQRT')-np.array((1,1))) < 0.01))
                
        self.assertTrue(np.all(discrete.information_mutual_normalised(np.array(((1,2,1), (1,2,1))))==1))
        self.assertTrue(np.all(np.abs(discrete.information_mutual_normalised(np.array(((2,2,1,1), (2,1,1,1))), norm_factor='X')-np.array(((1,0.31), (0.38,1)))) < 0.01))
        self.assertTrue(np.all(np.abs(discrete.information_mutual_normalised(np.array(((2,2,1,1), (2,1,1,1))), norm_factor='Y')-np.array(((1,0.38), (0.31,1)))) < 0.01))        
        self.assertTrue(np.all(np.abs(discrete.information_mutual_normalised(np.array(((2,2,1,1), (2,1,1,1))), norm_factor='X+Y')-np.array(((0.5,0.17), (0.17,0.5)))) < 0.01))
        self.assertTrue(np.all(np.abs(discrete.information_mutual_normalised(np.array(((2,2,1,1), (2,1,1,1))), norm_factor='MIN')-np.array(((1,0.38), (0.38,1)))) < 0.01))
        self.assertTrue(np.all(np.abs(discrete.information_mutual_normalised(np.array(((2,2,1,1), (2,1,1,1))), norm_factor='MAX')-np.array(((1,0.31), (0.31,1)))) < 0.01))        
        self.assertTrue(np.all(np.abs(discrete.information_mutual_normalised(np.array(((2,2,1,1), (2,1,1,1))), norm_factor='SQRT')-np.array(((1,0.35), (0.35,1)))) < 0.01))
                
        with np.errstate(invalid='ignore', divide='ignore'):       
            for S in ('X', 'Y', 'X+Y', 'MIN', 'MAX', 'XY', 'SQRT'):
                self.assertTrue(discrete.information_mutual_normalised(np.array(1),np.array(1), S).shape == tuple())
                self.assertTrue(discrete.information_mutual_normalised(np.array((1,)),np.array((1,)), S).shape == tuple())
                self.assertTrue(discrete.information_mutual_normalised(np.ones((1,)),np.ones((1,)), S).shape == tuple())
                self.assertTrue(discrete.information_mutual_normalised(np.ones((1,1)),np.ones((1,1)), S).shape == (1,))
                #Tests using cartesian_product=True
                self.assertTrue(discrete.information_mutual_normalised(np.array(1),np.array(1), S,True).shape == tuple())
                self.assertTrue(discrete.information_mutual_normalised(np.array((1,)),np.array((1,)), S,True).shape == tuple())
                self.assertTrue(discrete.information_mutual_normalised(np.ones((1,)),np.ones((1,)), S,True).shape == tuple())
                self.assertTrue(discrete.information_mutual_normalised(np.ones((1,1)),np.ones((1,1)), S,True).shape == (1,1))
                self.assertTrue(discrete.information_mutual_normalised(np.ones((1,)),np.ones((1,1)), S,True).shape == (1,))
                self.assertTrue(discrete.information_mutual_normalised(np.ones((1,1)),np.ones((1,)), S,True).shape == (1,))
                self.assertTrue(discrete.information_mutual_normalised(np.ones((1,1)),np.ones((1,1)), S,True).shape == (1,1))        
                
        with self.assertRaises(ValueError):
            discrete.information_mutual_normalised(np.array(()), np.array((1,2,3)))
        with self.assertRaises(ValueError):
            discrete.information_mutual_normalised(np.array((1,2,3)), np.array(()))            
        with self.assertRaises(ValueError):
            discrete.information_mutual_normalised(np.array((np.nan, 1, 2)), np.array((1,2,3)))
        with self.assertRaises(ValueError):
            discrete.information_mutual_normalised(np.array((1,2,3)), np.array((1,2,np.nan)))
        with self.assertRaises(ValueError):
            discrete.information_mutual_normalised(np.array((1,2,3)), np.array((1,2,3)), Alphabet_X=(np.nan,1,2,3), Alphabet_Y=(1,2,3))
        with self.assertRaises(ValueError):
            discrete.information_mutual_normalised(np.array((1,2,3)), np.array((1,2,3)), Alphabet_X=(1,2,3), Alphabet_Y=(np.nan,1,2,3))
        with self.assertRaises(ValueError):
            discrete.information_mutual_normalised(np.array((1,2,3)), np.array((1,2,3)), fill_value=np.nan)            
        with self.assertRaises(ValueError):            
            discrete.information_mutual_normalised(np.array((1,2,3)), np.array((4,1,2,3)))
        with self.assertRaises(ValueError):
            discrete.information_mutual_normalised(np.array((1,2,3)), np.array((1,2,3)), 3)            
        with self.assertRaises(ValueError):
            discrete.information_mutual_normalised(np.array(((1,2,3,4),(4,5,6,8))), np.array((4,1,2,3)))
        try:
            discrete.information_mutual_normalised(np.array(((1,2,3),(4,5,6))), np.array((4,1,2)), cartesian_product=True)
        except ValueError:
            self.fail("raised ValueError unexpectedly")
        #NB: No test for base here
        
        #
        # Tests using missing data (basic)
        #
        #Immutability test        
        X1 = np.random.randint(16, size=(10,10)) * 1.0
        X1[0,0] = -1
        X1_copy = np.copy(X1)
        X2 = np.random.randint(16, size=(10,10))
        X2[0,1] = -1
        X2_copy = np.copy(X2)
        discrete.information_mutual_normalised(X1,X2, fill_value=-1)
        self.assertTrue(np.all(X1_copy == X1) and np.all(X2_copy == X2))
        self.assertFalse(np.all(discrete.information_mutual_normalised(X1,X2, fill_value=-1) == discrete.information_mutual_normalised(X1,X2, fill_value=None)))
        
        # Tests using missing data involving masked arrays
        X1 = np.random.randint(16, size=(10,10)) * 1.0
        X1[0,0] = -1
        Mask1 = np.zeros_like(X1)
        Mask1[0,0] = 1
        X2 = np.random.randint(16, size=(10,10)) * 1.0
        X2[0,1] = -1
        Mask2 = np.zeros_like(X2)
        Mask2[0,1] = 1
        self.assertTrue(np.all(discrete.information_mutual_normalised(X1, X2, fill_value=-1) == discrete.information_mutual_normalised(np.ma.array(X1,mask=Mask1), X2, fill_value=-1)))
        self.assertTrue(np.all(discrete.information_mutual_normalised(X1, X2, fill_value=-1) == discrete.information_mutual_normalised(X1, np.ma.array(X2,mask=Mask2), fill_value=-1)))
        self.assertTrue(np.all(discrete.information_mutual_normalised(X1, X2, fill_value=-1) == discrete.information_mutual_normalised(np.ma.array(X1,mask=Mask1), np.ma.array(X2,mask=Mask2), fill_value=-1)))        
        
        ### Tests involving Alphabet and non-ML estimation
        ###
        ###
        #Alphabet immutability
        for norm_factor in ('Y','X','X+Y','Y+X','MIN','MAX','XY','YX','SQRT'):
            X1 = np.random.randint(16, size=(10,10))
            X2 = np.copy(X1)
            discrete.information_mutual_normalised(X2, X2, norm_factor, estimator=1, Alphabet_X=X2, Alphabet_Y=X2)        
            self.assertTrue(np.all(X2 == X1))
            discrete.information_mutual_normalised(X2, None, norm_factor, estimator=1, Alphabet_X=X2)
            self.assertTrue(np.all(X2 == X1))
        #No explicit alphabet
        for norm_factor in ('Y','X','X+Y','MIN','MAX','XY','SQRT'):
            Numerator = discrete.entropy((1,2,1,2), estimator=1, base=3)-discrete.entropy_conditional((1,2,1,2),(2,2,2,1), estimator=1, base=3)
            if norm_factor == 'Y':
                Denominator = discrete.entropy((2,2,2,1), estimator=1, base=3)
            elif norm_factor == 'X':
                Denominator = discrete.entropy((1,2,1,2), estimator=1, base=3)
            elif norm_factor == 'X+Y':
                Denominator = discrete.entropy((2,2,2,1), estimator=1, base=3) + discrete.entropy((1,2,1,2), estimator=1, base=3)
            elif norm_factor == 'MIN':
                Denominator = min(discrete.entropy((2,2,2,1), estimator=1, base=3),discrete.entropy((1,2,1,2), estimator=1, base=3))
            elif norm_factor == 'MAX':
                Denominator = max(discrete.entropy((2,2,2,1), estimator=1, base=3),discrete.entropy((1,2,1,2), estimator=1, base=3))
            elif norm_factor == 'XY':
                Denominator = discrete.entropy_joint(np.array(((2,2,2,1),(1,2,1,2))), estimator=1, base=3)
            elif norm_factor == 'SQRT':
                Denominator = np.sqrt(discrete.entropy((2,2,2,1), estimator=1, base=3) * discrete.entropy((1,2,1,2), estimator=1, base=3))                
            else:
                raise ValueError("Unsupported argument")
            self.assertTrue(np.allclose(discrete.information_mutual_normalised((1,2,1,2),(2,2,2,1), norm_factor, estimator=1, Alphabet_X=None, Alphabet_Y=None), Numerator / Denominator))
        #No explicit alphabet -- Cartesian product
        for norm_factor in ('Y','X','X+Y','MIN','MAX','XY','SQRT'):
            self.assertTrue(np.all(discrete.information_mutual_normalised(np.array(((1,2,1,2),(2,2,2,1))), None, norm_factor, estimator=1, Alphabet_X=None) != discrete.information_mutual_normalised(np.array(((1,2,1,2),(2,2,2,1))), None, norm_factor, Alphabet_X=None)))        
        #Larger alphabet
        for norm_factor in ('Y','X','X+Y','MIN','MAX','XY','SQRT'):            
            Numerator1 = discrete.entropy((1,2,1,2), estimator=1, Alphabet_X=None, base=3)-discrete.entropy_conditional((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=None, Alphabet_Y=(1,2,3), base=3)
            Numerator2 = discrete.entropy((1,2,1,2), estimator=1, Alphabet_X=(1,2,3), base=3)-discrete.entropy_conditional((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=(1,2,3), Alphabet_Y=None, base=3)
            Numerator3 = discrete.entropy((1,2,1,2), estimator=1, Alphabet_X=(1,2,3), base=3)-discrete.entropy_conditional((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=(1,2,3), Alphabet_Y=(1,2,3), base=3)
            if norm_factor == 'Y':
                Denominator1 = discrete.entropy((2,2,2,1), estimator=1, Alphabet_X=(1,2,3), base=3)
                Denominator2 = discrete.entropy((2,2,2,1), estimator=1, Alphabet_X=None, base=3)
                Denominator3 = discrete.entropy((2,2,2,1), estimator=1, Alphabet_X=(1,2,3), base=3)                
            elif norm_factor == 'X':
                Denominator1 = discrete.entropy((1,2,1,2), estimator=1, Alphabet_X=None, base=3)
                Denominator2 = discrete.entropy((1,2,1,2), estimator=1, Alphabet_X=(1,2,3), base=3)
                Denominator3 = discrete.entropy((1,2,1,2), estimator=1, Alphabet_X=(1,2,3), base=3)                
            elif norm_factor == 'X+Y':
                Denominator1 = discrete.entropy((2,2,2,1), estimator=1, Alphabet_X=(1,2,3), base=3) + discrete.entropy((1,2,1,2), estimator=1, Alphabet_X=None, base=3)
                Denominator2 = discrete.entropy((2,2,2,1), estimator=1, Alphabet_X=None, base=3) + discrete.entropy((1,2,1,2), estimator=1, Alphabet_X=(1,2,3), base=3)
                Denominator3 = discrete.entropy((2,2,2,1), estimator=1, Alphabet_X=(1,2,3), base=3) + discrete.entropy((1,2,1,2), estimator=1, Alphabet_X=(1,2,3), base=3)                
            elif norm_factor == 'MIN':
                Denominator1 = min(discrete.entropy((2,2,2,1), Alphabet_X=(1,2,3), estimator=1, base=3),discrete.entropy((1,2,1,2), estimator=1, Alphabet_X=None, base=3))
                Denominator2 = min(discrete.entropy((2,2,2,1), Alphabet_X=None, estimator=1, base=3),discrete.entropy((1,2,1,2), estimator=1, Alphabet_X=(1,2,3), base=3))
                Denominator3 = min(discrete.entropy((2,2,2,1), Alphabet_X=(1,2,3), estimator=1, base=3),discrete.entropy((1,2,1,2), estimator=1, Alphabet_X=(1,2,3), base=3))
            elif norm_factor == 'MAX':
                Denominator1 = max(discrete.entropy((2,2,2,1), estimator=1, Alphabet_X=(1,2,3), base=3),discrete.entropy((1,2,1,2), estimator=1, Alphabet_X=None, base=3))
                Denominator2 = max(discrete.entropy((2,2,2,1), estimator=1, Alphabet_X=None, base=3),discrete.entropy((1,2,1,2), estimator=1, Alphabet_X=(1,2,3), base=3))
                Denominator3 = max(discrete.entropy((2,2,2,1), estimator=1, Alphabet_X=(1,2,3), base=3),discrete.entropy((1,2,1,2), estimator=1, Alphabet_X=(1,2,3), base=3))                
            elif norm_factor == 'XY':
                Denominator1 = discrete.entropy_joint(np.array(((2,2,2,1),(1,2,1,2))), estimator=1, Alphabet_X=np.array(((1,2,-1),(1,2,3))), base=3)
                Denominator2 = discrete.entropy_joint(np.array(((2,2,2,1),(1,2,1,2))), estimator=1, Alphabet_X=np.array(((1,2,3),(1,2,-1))), base=3)
                Denominator3 = discrete.entropy_joint(np.array(((2,2,2,1),(1,2,1,2))), estimator=1, Alphabet_X=np.array(((1,2,3),(1,2,3))), base=3)               
            elif norm_factor == 'SQRT':
                Denominator1 = np.sqrt(discrete.entropy((2,2,2,1), Alphabet_X=(1,2,3), estimator=1, base=3) * discrete.entropy((1,2,1,2), Alphabet_X=None, estimator=1, base=3))
                Denominator2 = np.sqrt(discrete.entropy((2,2,2,1), Alphabet_X=None, estimator=1, base=3) * discrete.entropy((1,2,1,2), Alphabet_X=(1,2,3), estimator=1, base=3))
                Denominator3 = np.sqrt(discrete.entropy((2,2,2,1), Alphabet_X=(1,2,3), estimator=1, base=3) * discrete.entropy((1,2,1,2), Alphabet_X=(1,2,3), estimator=1, base=3))                
            else:
                raise ValueError("Unsupported argument")
            self.assertTrue(np.allclose(discrete.information_mutual_normalised((1,2,1,2),(2,2,2,1), norm_factor, estimator=1, Alphabet_X=None, Alphabet_Y=(1,2,3)), Numerator1 / Denominator1))
            self.assertTrue(np.allclose(discrete.information_mutual_normalised((1,2,1,2),(2,2,2,1), norm_factor, estimator=1, Alphabet_X=(1,2,3), Alphabet_Y=None), Numerator2 / Denominator2))
            self.assertTrue(np.allclose(discrete.information_mutual_normalised((1,2,1,2),(2,2,2,1), norm_factor, estimator=1, Alphabet_X=(1,2,3), Alphabet_Y=(1,2,3)), Numerator3 / Denominator3))        
        #Larger alphabet -- Cartesian product
        for norm_factor in ('Y','X','X+Y','MIN','MAX','XY','SQRT'):        
            self.assertTrue(np.all(discrete.information_mutual_normalised(np.array(((1,2,1,2),(2,2,2,1))), None, norm_factor, estimator=1, Alphabet_X=np.array(((1,2,3),(1,2,3)))) != discrete.information_mutual_normalised(np.array(((2,2,2,1),(2,2,2,1))), None, norm_factor, Alphabet_X=np.array(((1,2,3),(1,2,3))))))
        for norm_factor in ('Y','X','X+Y','MIN','MAX','XY','SQRT'):
            #Empty alphabet
            with self.assertRaises(ValueError):
                discrete.information_mutual_normalised([1, 2], [1, 2], norm_factor, estimator=1, Alphabet_X=np.array(()), Alphabet_Y=None)
            with self.assertRaises(ValueError):
                discrete.information_mutual_normalised([1, 2], [1, 2], norm_factor, estimator=1, Alphabet_X=None, Alphabet_Y=np.array(()))
            with self.assertRaises(ValueError):
                discrete.information_mutual_normalised([1, 2], None, norm_factor, estimator=1, Alphabet_X=np.array(()))            
            #Smaller alphabet
            with self.assertRaises(ValueError):
                discrete.information_mutual_normalised([1,2], [1,2], norm_factor, estimator=1, Alphabet_X=[1,2], Alphabet_Y=[1,1])
            with self.assertRaises(ValueError):
                discrete.information_mutual_normalised([1,2], [1,2], norm_factor, estimator=1, Alphabet_X=[1,1], Alphabet_Y=[1,2])        
            with self.assertRaises(ValueError):
                discrete.information_mutual_normalised(np.array(((1,2),(1,2))), None, norm_factor, estimator=1, Alphabet_X=np.array(((1,2),(1,-1))))
            #Alphabet with incorrect dimensions
            with self.assertRaises(ValueError):
                discrete.information_mutual_normalised(np.array(((1,2),(1,2))), np.array(((1,2),(1,2))), norm_factor, estimator=1, Alphabet_X=np.array((1,2)), Alphabet_Y=np.array(((1,2),(1,2))))
            with self.assertRaises(ValueError):
                discrete.information_mutual_normalised(np.array(((1,2),(1,2))), np.array(((1,2),(1,2))), norm_factor, estimator=1, Alphabet_X=np.array(((1,2),(1,2))), Alphabet_Y=np.array((1,2)))            
            with self.assertRaises(ValueError):
                discrete.information_mutual_normalised(np.array(((1,2),(1,2))), None, norm_factor, estimator=1, Alphabet_X=np.array((1,2)))        
        
    def test_information_lautum(self):
        #Immutability test
        X1 = np.random.randint(16, size=(10,10))
        X1_copy = np.copy(X1)
        X2 = np.random.randint(16, size=(10,10))
        X2_copy = np.copy(X2)        
        discrete.information_lautum(X1,X2)
        self.assertTrue(np.all(X1_copy == X1) and np.all(X2_copy == X2))
        
        #Example from Palomar and Verdu (2006), p. 3:
        D = np.zeros((2,100))
        D[:,96:98] = 1
        D[0,98] = 1
        D[1,99] = 1
        X = D[0]
        Y = D[1]
        
        self.assertTrue(discrete.information_lautum(np.array((1.0, 1.0, 1.0, 1.0)), np.array((1.0, 1.0, 1.0, 1.0)), base=2) == 0)
        self.assertTrue(np.abs(discrete.information_lautum(X, Y, base=2) - 0.0584) < 0.0001)
        self.assertTrue(discrete.information_lautum(np.array((2,2,1,1,1)), np.array((2,1,1,1,2)), base=2) == discrete.divergence_kullbackleibler_pmf(np.array((0.16, 0.24, 0.36, 0.24)), np.array((0.2, 0.2, 0.4,0.2))))
        self.assertTrue(np.allclose(discrete.information_lautum(np.array((2,2,1,1,1)), np.array((2,1,1,1,2)), base=np.exp(1)), discrete.divergence_kullbackleibler_pmf(np.array((0.16, 0.24, 0.36, 0.24)), np.array((0.2, 0.2, 0.4,0.2)), base=np.exp(1))))
        #Tests using cartesian_product=True
        self.assertTrue(discrete.information_lautum(np.array((1.0, 1.0, 1.0, 1.0)), np.array((1.0, 1.0, 1.0, 1.0)), cartesian_product=True, base=2) == 0)
        self.assertTrue(np.abs(discrete.information_lautum(X, Y, cartesian_product=True, base=2) - 0.0584) < 0.0001)
        self.assertTrue(discrete.information_lautum(np.array((2,2,1,1,1)), np.array((2,1,1,1,2)), cartesian_product=True, base=2) == discrete.divergence_kullbackleibler_pmf(np.array((0.16, 0.24, 0.36, 0.24)), np.array((0.2, 0.2, 0.4,0.2))))
        self.assertTrue(np.allclose(discrete.information_lautum(np.array((2,2,1,1,1)), np.array((2,1,1,1,2)), cartesian_product=True, base=np.exp(1)), discrete.divergence_kullbackleibler_pmf(np.array((0.16, 0.24, 0.36, 0.24)), np.array((0.2, 0.2, 0.4,0.2)), base=np.exp(1))))        
        
        self.assertTrue(discrete.information_lautum(np.ones(5),np.ones(5))==0)
        self.assertTrue(discrete.information_lautum(np.ones((5,1)).T,np.ones((5,1)).T)==0)
        self.assertTrue(discrete.information_lautum(np.ones((5,1)),np.ones((5,1))).size==5)
        self.assertTrue(np.all(discrete.information_lautum(np.ones((5,1)),np.ones((5,1)))==0))
        #Tests using cartesian_product=True        
        self.assertTrue(discrete.information_lautum(np.ones(5),np.ones(5), cartesian_product=True)==0)
        self.assertTrue(discrete.information_lautum(np.ones((5,1)).T,np.ones((5,1)).T, cartesian_product=True)==0)
        self.assertTrue(discrete.information_lautum(np.ones((5,1)),np.ones((5,1)), cartesian_product=True).size==25)
        self.assertTrue(np.all(discrete.information_lautum(np.ones((5,1)),np.ones((5,1)), cartesian_product=True)==0))
        
        self.assertTrue(discrete.information_lautum(np.array(4),np.array(4)) == 0)
        self.assertTrue(discrete.information_lautum(np.array((4,)),np.array((4,))) == 0)
        self.assertTrue(discrete.information_lautum(np.ones((1,)),np.ones((1,))) == 0)
        self.assertTrue(discrete.information_lautum(np.ones((1,1)),np.ones((1,1))) == 0)        
        #Tests using cartesian_product=True        
        self.assertTrue(discrete.information_lautum(np.array(4),np.array(4), cartesian_product=True) == 0)
        self.assertTrue(discrete.information_lautum(np.array((4,)),np.array((4,)), cartesian_product=True) == 0)
        self.assertTrue(discrete.information_lautum(np.ones((1,)),np.ones((1,)), cartesian_product=True) == 0)
        self.assertTrue(discrete.information_lautum(np.ones((1,1)),np.ones((1,1)), cartesian_product=True) == 0)        
        
        self.assertTrue(discrete.information_lautum(np.array(1),np.array(1)).shape == tuple())
        self.assertTrue(discrete.information_lautum(np.array((1,)),np.array((1,))).shape == tuple())
        self.assertTrue(discrete.information_lautum(np.ones((1,)),np.ones((1,))).shape == tuple())
        self.assertTrue(discrete.information_lautum(np.ones((1,1)),np.ones((1,1))).shape == (1,))
        #Tests using cartesian_product=True
        self.assertTrue(discrete.information_lautum(np.array(1),np.array(1),True).shape == tuple())
        self.assertTrue(discrete.information_lautum(np.array((1,)),np.array((1,)),True).shape == tuple())
        self.assertTrue(discrete.information_lautum(np.ones((1,)),np.ones((1,)),True).shape == tuple())
        self.assertTrue(discrete.information_lautum(np.ones((1,1)),np.ones((1,1)),True).shape == (1,1))
        self.assertTrue(discrete.information_lautum(np.ones((1,)),np.ones((1,1)),True).shape == (1,))
        self.assertTrue(discrete.information_lautum(np.ones((1,1)),np.ones((1,)),True).shape == (1,))
        self.assertTrue(discrete.information_lautum(np.ones((1,1)),np.ones((1,1)),True).shape == (1,1))                
        
        self.assertTrue(np.all(discrete.information_lautum(np.array(((1,2,2,1), (1,2,1,2))))==np.array(((np.inf,0),(0,np.inf)))))
        self.assertTrue(np.all(discrete.information_lautum(np.array(((1,2,2,1), (1,2,1,2))).T,np.array(((1,2,2,1), (1,2,1,2))).T)==np.array((0,0,np.inf,np.inf))))
        #Tests using cartesian_product=True
        self.assertTrue(np.all(discrete.information_lautum(np.array(((1,2,2,1), (1,2,1,2))), cartesian_product=True)==np.array(((np.inf,0),(0,np.inf)))))
        H = discrete.information_lautum(np.array(((1,2,2,1), (1,2,1,2))).T,np.array(((1,2,2,1), (1,2,1,2))).T, cartesian_product=True)
        self.assertTrue(np.all(H[0:2] == 0) and np.all(H[:,0:2] == 0) and np.all(H[2:4,2:4] == np.inf))
        
        with self.assertRaises(ValueError):
            discrete.information_lautum(np.array(()), np.array((1,2,3)))
        with self.assertRaises(ValueError):
            discrete.information_lautum(np.array((1,2,3)), np.array(()))            
        with self.assertRaises(ValueError):
            discrete.information_lautum(np.array((np.nan,1,2)), np.array((1,2,3)))
        with self.assertRaises(ValueError):
            discrete.information_lautum(np.array((1,1,2)), np.array((1,2,np.nan)))
        with self.assertRaises(ValueError):
            discrete.information_lautum(np.array((1,2,3)), np.array((1,2,3)), Alphabet_X=(np.nan,1,2,3), Alphabet_Y=(1,2,3))
        with self.assertRaises(ValueError):
            discrete.information_lautum(np.array((1,2,3)), np.array((1,2,3)), Alphabet_X=(1,2,3), Alphabet_Y=(np.nan,1,2,3))
        with self.assertRaises(ValueError):
            discrete.information_lautum(np.array((1,1,2)), np.array((1,1,2)), fill_value=np.nan)
        with self.assertRaises(ValueError):
            discrete.information_lautum(np.array(((1,2,3),(4,5,6))), np.array((4,1,2)))
        try:
            discrete.information_lautum(np.array(((1,2,3),(4,5,6))), np.array((4,1,2)), True)
            discrete.information_lautum(np.array(((1,2,3),(4,5,6))))
        except ValueError:
            self.fail("raised ValueError unexpectedly")
        with self.assertRaises(ValueError):
            discrete.information_lautum(np.array((2, 1)), base=-1)
            
        #
        # Tests using missing data
        #
        #Immutability test
        X1 = np.random.randint(16, size=(10,10)) * 1.0
        X1[0,0] = -1
        X1_copy = np.copy(X1)
        X2 = np.random.randint(16, size=(10,10))
        X2[0,1] = -1
        X2_copy = np.copy(X2)
        discrete.information_lautum(X1,X2, fill_value=-1)
        self.assertTrue(np.all(X1_copy == X1) and np.all(X2_copy == X2))
        
        # Tests using missing data involving masked arrays
        X1 = np.random.randint(16, size=(10,10)) * 1.0
        X1[0,0] = -1
        Mask1 = np.zeros_like(X1)
        Mask1[0,0] = 1
        X2 = np.random.randint(16, size=(10,10)) * 1.0
        X2[0,1] = -1
        Mask2 = np.zeros_like(X2)
        Mask2[0,1] = 1
        self.assertTrue(np.all(discrete.information_lautum(X1, X2, fill_value=-1) == discrete.information_lautum(np.ma.array(X1,mask=Mask1), X2, fill_value=-1)))
        self.assertTrue(np.all(discrete.information_lautum(X1, X2, fill_value=-1) == discrete.information_lautum(X1, np.ma.array(X2,mask=Mask2), fill_value=-1)))
        self.assertTrue(np.all(discrete.information_lautum(X1, X2, fill_value=-1) == discrete.information_lautum(np.ma.array(X1,mask=Mask1), np.ma.array(X2,mask=Mask2), fill_value=-1)))

        #Tests involving masked arrays
        self.assertTrue(discrete.information_lautum(np.ma.array(((4,4,2,2,2)), mask=((0,0,1,1,1))), np.ma.array(((4,4,2,4,2)), mask=((0,0,1,1,1)))) == 0)
        self.assertTrue(np.isnan(discrete.information_lautum(np.ma.array(((4,4,2,2,2)), mask=((1,1,1,1,1))), np.ma.array(((4,4,2,4,2)), mask=((0,0,1,1,1))))))
        self.assertTrue(np.isnan(discrete.information_lautum(np.ma.array(((4,4,2,2,2)), mask=((0,0,1,1,1))), np.ma.array(((4,4,2,4,2)), mask=((1,1,1,1,1))))))

        self.assertTrue(discrete.information_lautum(np.array(((4,4,-1,-1,-1))), np.array(((4,4,-1,-1,-1))), fill_value=-1) == 0)
        self.assertTrue(np.isnan(discrete.information_lautum(np.array(((-1,-1,-1,-1,-1))), np.array(((4,4,-1,-1,-1))), fill_value=-1)))
        self.assertTrue(np.isnan(discrete.information_lautum(np.array(((4,4,-1,-1,-1))), np.array(((-1,-1,-1,-1,-1))), fill_value=-1)))
        
        with self.assertRaises(ValueError):
            discrete.information_lautum(np.ma.array(((4,4,2,2,2)), mask=((0,0,1,1,1))), np.array(((1,2,3))), fill_value=-1)
            
        ### Tests involving Alphabet and non-ML estimation
        ###
        ###
        #Alphabet immutability 
        X1 = np.random.randint(16, size=(10,10))
        X2 = np.copy(X1)
        discrete.information_lautum(X2, X2, estimator=1, Alphabet_X=X2, Alphabet_Y=X2)
        self.assertTrue(np.all(X2 == X1))
        discrete.information_lautum(X2, estimator=1, Alphabet_X=X2)
        self.assertTrue(np.all(X2 == X1))        
        #No explicit alphabet
        self.assertTrue(discrete.information_lautum((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=(1,2), Alphabet_Y=None) == discrete.information_lautum((1,1,1,1,2,2,2,2),(1,2,2,2,1,1,2,2), Alphabet_X=None, Alphabet_Y=None))
        self.assertTrue(discrete.information_lautum((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=None, Alphabet_Y=(1,2)) == discrete.information_lautum((1,1,1,1,2,2,2,2),(1,2,2,2,1,1,2,2), Alphabet_X=None, Alphabet_Y=None))
        self.assertTrue(discrete.information_lautum((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=(1,2), Alphabet_Y=(1,2)) == discrete.information_lautum((1,1,1,1,2,2,2,2),(1,2,2,2,1,1,2,2), Alphabet_X=None, Alphabet_Y=None))
        #No explicit alphabet -- Cartesian product
        A = discrete.information_lautum(np.array(((1,2,1,2),(2,2,2,1))), estimator=1, Alphabet_X=None, base=3)
        B = discrete.information_lautum(np.array(((1,1,1,1,2,2,2,2),(1,2,2,2,1,1,2,2))), estimator=0, Alphabet_X=None, base=3)        
        self.assertTrue(np.all(A[1,0]==B[1,0]) and np.all(A[0,1]==B[0,1]))
        #Larger alphabet
        self.assertTrue(discrete.information_lautum((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=(1,2,3), Alphabet_Y=None) == discrete.information_lautum((1,1,1,1,2,2,2,2,3,3),(1,2,2,2,1,1,2,2,1,2), Alphabet_X=None, Alphabet_Y=None))
        self.assertTrue(discrete.information_lautum((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=None, Alphabet_Y=(1,2,3)) == discrete.information_lautum((1,1,1,1,2,2,2,2,1,2),(1,2,2,2,1,1,2,2,3,3), Alphabet_X=None, Alphabet_Y=None))
        self.assertTrue(discrete.information_lautum((1,2,1,2),(2,2,2,1), estimator=1, Alphabet_X=(1,2,3), Alphabet_Y=(1,2,3)) == discrete.information_lautum((1,1,1,1,2,2,2,2,3,3,3,1,2),(1,2,2,2,1,1,2,2,1,2,3,3,3), Alphabet_X=None, Alphabet_Y=None))        
        #Larger alphabet -- Cartesian product
        A = discrete.information_lautum(np.array(((1,2),(2,2))), estimator=1, Alphabet_X=np.array(((1,2,3),(1,2,3))), base=3)
        B = discrete.information_lautum(np.array(((1,1,1,1,2,2,2,2,3,3,3),(1,2,2,3,1,2,2,3,1,2,3))), estimator=0, Alphabet_X=None, base=3)
        self.assertTrue(np.all(A[1,0]==B[1,0]) and np.all(A[0,1]==B[0,1]))
        #Empty alphabet
        with self.assertRaises(ValueError):
            discrete.information_lautum([1, 2], [1, 2], estimator=1, Alphabet_X=np.array(()), Alphabet_Y=None, base=3)
        with self.assertRaises(ValueError):
            discrete.information_lautum([1, 2], [1, 2], estimator=1, Alphabet_X=None, Alphabet_Y=np.array(()), base=3)
        with self.assertRaises(ValueError):
            discrete.information_lautum([1, 2], estimator=1, Alphabet_X=np.array(()), base=3)            
        #Smaller alphabet
        with self.assertRaises(ValueError):
            discrete.information_lautum([1,2], [1,2], estimator=1, Alphabet_X=[1,2], Alphabet_Y=[1,1], base=3)
        with self.assertRaises(ValueError):
            discrete.information_lautum([1,2], [1,2], estimator=1, Alphabet_X=[1,1], Alphabet_Y=[1,2], base=3)        
        with self.assertRaises(ValueError):
            discrete.information_lautum(np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array(((1,2),(1,-1))), base=3)
        #Alphabet with incorrect dimensions
        with self.assertRaises(ValueError):
            discrete.information_lautum(np.array(((1,2),(1,2))), np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array((1,2)), Alphabet_Y=np.array(((1,2),(1,2))), base=3)
        with self.assertRaises(ValueError):
            discrete.information_lautum(np.array(((1,2),(1,2))), np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array(((1,2),(1,2))), Alphabet_Y=np.array((1,2)), base=3)            
        with self.assertRaises(ValueError):
            discrete.information_lautum(np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array((1,2)), base=3)            
            
    def test_information_binding(self):
        #Immutability test
        X1 = np.random.randint(16, size=(10,10))
        X1_copy = np.copy(X1)
        discrete.information_binding(X1)
        self.assertTrue(np.all(X1_copy == X1))
                
        self.assertTrue(discrete.information_binding(np.array((1.0, 1.0, 1.0, 1.0)), base=2) == 0)
        self.assertTrue(discrete.information_binding(np.array((1.0, 1.0, 2.0, 2.0)), base=2) == 0)
        self.assertTrue(discrete.information_binding(np.array(((1,1,2,2),(2,2,2,2))), base=2) == 0)
        self.assertTrue(discrete.information_binding(np.array(((1,1,2,2),(1,1,2,2))), base=2) == 1)
        self.assertTrue(discrete.information_binding(np.array(((2,1,2,2),(1,1,2,2))), base=2) == discrete.entropy_joint(np.array(((2,1,1,1),(1,1,2,2)))) - discrete.entropy_conditional([1,1,2,2],[2,1,1,1]) - discrete.entropy_conditional([2,1,1,1],[1,1,2,2]))        
        
        self.assertTrue(discrete.information_binding(np.ones(5))==0)
        self.assertTrue(discrete.information_binding(np.ones((5,1)).T)==0)
        self.assertTrue(discrete.information_binding(np.ones((5,1))).size==1)
        self.assertTrue(discrete.information_binding(np.ones((5,1)))==0)
        
        self.assertTrue(discrete.information_binding(np.array(4)) == 0)
        self.assertTrue(discrete.information_binding(np.array((4,))) == 0)
        self.assertTrue(discrete.information_binding(np.ones((1,))) == 0)
        self.assertTrue(discrete.information_binding(np.ones((1,1))) == 0)
        
        self.assertTrue(discrete.information_binding(np.array(1)).shape == tuple())
        self.assertTrue(discrete.information_binding(np.array((1,))).shape == tuple())
        self.assertTrue(discrete.information_binding(np.ones((1,))).shape == tuple())
        self.assertTrue(discrete.information_binding(np.ones((1,1))).shape == tuple())
        
        with self.assertRaises(ValueError):
            discrete.information_binding(np.array(()))
        with self.assertRaises(ValueError):
            discrete.information_binding(np.array((np.nan,1,2)))
        with self.assertRaises(ValueError):
            discrete.information_binding(np.array((1,2)), Alphabet_X=np.array((1,2,np.nan)))
        with self.assertRaises(ValueError):
            discrete.information_binding(np.array((2,1)), fill_value=np.nan)
        with self.assertRaises(ValueError):
            discrete.information_binding(np.array((2,1)), base=-1)
            
        #
        # Tests using missing data (basic)
        #
        #Immutability test        
        X1 = np.random.randint(16, size=(10,10)) * 1.0
        X1[0,0] = -1
        X1_copy = np.copy(X1)
        discrete.information_binding(X1, fill_value=-1)
        self.assertTrue(np.all(X1_copy == X1))
        self.assertFalse(np.all(discrete.information_binding(X1,fill_value=0) == discrete.information_binding(X1, fill_value=None)))
        
        # Tests using missing data involving masked arrays
        X1 = np.random.randint(16, size=(10,10)) * 1.0
        X1[0,0] = -1
        Mask = np.zeros_like(X1)
        Mask[0,0] = 1
        self.assertTrue(discrete.information_binding(X1, fill_value=-1) == discrete.information_binding(np.ma.array(X1,mask=Mask)))        
        
        # Tests using missing data and None values
        self.assertTrue(discrete.information_binding(np.array(((1,2,1,2,1,1,None,None),(1,1,1,2,1,1,1,1))), fill_value=-1) == discrete.information_binding(np.array(((1,2,1,2,1,1,3,3),(1,1,1,2,1,1,1,1)))))
        self.assertTrue(discrete.information_binding(np.array(((1,2,1,2,1,1,None,None),(1,1,1,2,1,1,1,1))), fill_value=None) == - discrete.entropy_joint(np.array(((1,2,1,2,1,1),(1,1,1,2,1,1)))) + discrete.entropy((1,2,1,2,1,1)) + discrete.entropy((1,1,1,2,1,1,1,1)))
        self.assertTrue(np.allclose(discrete.information_binding(np.ma.array(((1,2,1,2,1,1,None,None),(1,1,1,2,1,1,1,1)),mask=((0,0,0,0,0,0,1,1),(0,0,0,0,0,0,0,0)))), - discrete.entropy_joint(np.array(((1,2,1,2,1,1),(1,1,1,2,1,1)))) + discrete.entropy((1,2,1,2,1,1)) + discrete.entropy((1,1,1,2,1,1,1,1))))
        self.assertTrue(discrete.information_binding(np.array(((1,2,1,2,1,1,None,None),(1,1,1,2,1,1,1,1))), Alphabet_X=((1,2,None),(1,2,-1)), fill_value=-1) == discrete.information_binding(np.array(((1,2,1,2,1,1,3,3),(1,1,1,2,1,1,1,1))), Alphabet_X=((1,2,3),(1,2,-1)), fill_value=-1))
        self.assertTrue(np.allclose(discrete.information_binding(np.array(((1,2,1,2,1,1,None,None),(1,1,1,2,1,1,1,1))), Alphabet_X=((1,2,None),(1,2,None)), fill_value=None), - discrete.entropy_joint(np.array(((1,2,1,2,1,1),(1,1,1,2,1,1)))) + discrete.entropy((1,2,1,2,1,1)) + discrete.entropy((1,1,1,2,1,1,1,1))))
        self.assertTrue(np.allclose(discrete.information_binding(np.array(((1,2,1,2,1,1),(1,1,1,2,1,1))), Alphabet_X=np.ma.array(((1,2,None),(1,2,None)),mask=((0,0,1),(0,0,0)))), discrete.information_binding(np.array(((1,2,1,2,1,1),(1,1,1,2,1,1))), Alphabet_X=np.array(((1,2,None),(1,2,3))))))
        self.assertTrue(discrete.information_binding(np.array(((1,2,1,2,1,1,None,None),(1,1,1,2,1,1,1,1))), Alphabet_X=np.array(((1,2,None),(1,2,None))), fill_value=-1) == discrete.information_binding(np.array(((1,2,1,2,1,1,3,3),(1,1,1,2,1,1,1,1))), Alphabet_X=np.array(((1,2,3),(1,2,3)))))
        self.assertTrue(np.allclose(discrete.information_binding(np.array(((1,2,1,2,1,1,None,None),(1,1,1,2,1,1,1,1))), Alphabet_X=((1,2,None),(1,2,None)), fill_value=None), - discrete.entropy_joint(np.array(((1,2,1,2,1,1),(1,1,1,2,1,1)))) + discrete.entropy((1,2,1,2,1,1)) + discrete.entropy((1,1,1,2,1,1,1,1))))
        self.assertTrue(np.allclose(discrete.information_binding(np.ma.array(((1,2,1,2,1,1,None,None),(1,1,1,2,1,1,1,1)),mask=((0,0,0,0,0,0,1,1),(0,0,0,0,0,0,0,0))), Alphabet_X=np.ma.array(((1,2,3),(1,2,3)),mask=((0,0,1),(0,0,0)))), - discrete.entropy_joint(np.array(((1,2,1,2,1,1),(1,1,1,2,1,1)))) + discrete.entropy((1,2,1,2,1,1)) + discrete.entropy((1,1,1,2,1,1,1,1))))
        
        ### Tests involving Alphabet and non-ML estimation
        ###
        ###
        #Alphabet immutability 
        X1 = np.random.randint(16, size=(10,10))
        X2 = np.copy(X1)
        discrete.information_binding(X2, estimator=1, Alphabet_X=X2)
        self.assertTrue(np.all(X2 == X1))
        #No explicit alphabet
        self.assertTrue(discrete.information_binding(np.array(((1,2,1),(1,2,2))),estimator=1,Alphabet_X=None,base=3) == discrete.entropy_joint(np.array(((1,2,1),(1,2,2))), estimator=1,Alphabet_X=None,base=3)-2*discrete.entropy_conditional([1,2,1],[1,2,2],estimator=1,base=3))
        #Larger alphabet
        self.assertTrue(np.abs(discrete.information_binding(np.array(((1,2,1),(1,2,2))),estimator=1,Alphabet_X=np.array(([1,2,3],[1,2,3])),base=3)- (-discrete.entropy_joint(np.array(((1,2,1),(1,2,2))),estimator=1,Alphabet_X=np.array(([1,2,3],[1,2,3])),base=3)+2*discrete.entropy([1,2,1],Alphabet_X=[1,2,3],estimator=1,base=3)))< 1E-03)
        #Empty alphabet
        with self.assertRaises(ValueError):
            discrete.information_binding([1, 2], estimator=1, Alphabet_X=np.array(()), base=3)
        #Smaller alphabet
        with self.assertRaises(ValueError):
            discrete.information_binding([1, 2], estimator=1, Alphabet_X=[1,], base=3)
        #Alphabet with incorrect dimensions
        with self.assertRaises(ValueError):
            discrete.information_binding(np.array((1,2)), estimator=1, Alphabet_X=np.array(((1,2),(1,2))), base=3)
        
    def test_information_mutual_conditional(self):
        #Immutability test
        X1 = np.random.randint(16, size=(10,10))
        X1_copy = np.copy(X1)
        X2 = np.random.randint(16, size=(10,10))
        X2_copy = np.copy(X2)
        X3 = np.random.randint(16, size=(10,10))
        X3_copy = np.copy(X3)
        discrete.information_mutual_conditional(X1,X2,X3)
        self.assertTrue(np.all(X1_copy == X1) and np.all(X2_copy == X2) and np.all(X3_copy == X3))
        discrete.information_mutual_conditional(X1,X2,X3,True)
        self.assertTrue(np.all(X1_copy == X1) and np.all(X2_copy == X2) and np.all(X3_copy == X3))        
        
        X = [1,1,2,2]
        Y = [1,2,2,2]
        Z = [1,1,1,2]
        self.assertTrue(discrete.information_mutual_conditional(X,Y,Z) == discrete.entropy_joint(np.vstack((X,Z))) + discrete.entropy_joint(np.vstack((Y,Z))) - discrete.entropy_joint(np.vstack((X,Y,Z))) - discrete.entropy(Z))
        #Tests using cartesian_product=True
        X = [1,1,2,2]
        Y = [1,2,2,2]
        Z = np.array(([1,1,1,2],[1,1,1,1]))
        self.assertTrue(np.allclose(discrete.information_mutual_conditional(X,Y,Z,True), np.array((discrete.entropy_joint(np.vstack((X,Z[0]))) + discrete.entropy_joint(np.vstack((Y,Z[0]))) - discrete.entropy_joint(np.vstack((X,Y,Z[0]))) - discrete.entropy(Z[0]), discrete.information_mutual(X,Y)))))
        X = [1,1,2,2]
        Y = np.array(([1,2,2,2],[1,1,1,2]))
        Z = [1,1,1,2]
        self.assertTrue(np.allclose(discrete.information_mutual_conditional(X,Y,Z,True), np.array((discrete.entropy_joint(np.vstack((X,Z))) + discrete.entropy_joint(np.vstack((Y[0],Z))) - discrete.entropy_joint(np.vstack((X,Y[0],Z))) - discrete.entropy(Z), 0), dtype=float)))
        X = np.array(([1,1,2,2], [1,1,1,2]))
        Y = [1,2,2,2]
        Z = [1,1,1,2]
        self.assertTrue(np.allclose(discrete.information_mutual_conditional(X,Y,Z,True), np.array((discrete.entropy_joint(np.vstack((X[0],Z))) + discrete.entropy_joint(np.vstack((Y,Z))) - discrete.entropy_joint(np.vstack((X[0],Y,Z))) - discrete.entropy(Z), 0), dtype=float)))        
        
        self.assertTrue(discrete.information_mutual_conditional(np.ones(5),np.ones(5),np.ones(5)).shape==tuple())
        self.assertTrue(discrete.information_mutual_conditional(np.ones((5,1)).T,np.ones((5,1)).T,np.ones((5,1)).T).shape==(1,))
        self.assertTrue(discrete.information_mutual_conditional(np.ones((5,1)),np.ones((5,1)),np.ones((5,1))).shape==(5,))
        self.assertTrue(discrete.information_mutual_conditional(np.ones((5,6,1)),np.ones((5,6,1)),np.ones((5,6,1))).shape==(5,6))        
        #Tests using cartesian_product=True
        self.assertTrue(discrete.information_mutual_conditional(np.ones(5),np.ones(5),np.ones(5),True).shape == tuple())
        self.assertTrue(discrete.information_mutual_conditional(np.ones((1,5)),np.ones(5),np.ones(5),True).shape == (1,))
        self.assertTrue(discrete.information_mutual_conditional(np.ones(5),np.ones((1,5)),np.ones(5),True).shape == (1,))
        self.assertTrue(discrete.information_mutual_conditional(np.ones(5),np.ones(5),np.ones((1,5)),True).shape == (1,))        
        self.assertTrue(discrete.information_mutual_conditional(np.ones(5),np.ones((1,5)),np.ones((1,5)),True).shape == (1,1))        
        self.assertTrue(discrete.information_mutual_conditional(np.ones((1,5)),np.ones((1,5)),np.ones((1,5)),True).shape == (1,1,1))
        self.assertTrue(discrete.information_mutual_conditional(np.ones((6,3,5)),np.ones((1,5)),np.ones((1,5)),True).shape == (6,3,1,1))
        self.assertTrue(discrete.information_mutual_conditional(np.ones((6,3,5)),np.ones((1,5)),np.ones((5)),True).shape == (6,3,1))        
                
        self.assertTrue(discrete.information_mutual_conditional(np.ones((5,1)).T,np.ones((5,1)).T,np.ones((5,1)).T)==0)
        self.assertTrue(discrete.information_mutual_conditional(np.ones((5,1)),np.ones((5,1)),np.ones((5,1))).size==5)
        self.assertTrue(np.all(discrete.information_mutual_conditional(np.ones((5,1)),np.ones((5,1)),np.ones((5,1)))==0))        
        
        self.assertTrue(discrete.information_mutual_conditional(np.array(4),np.array(4),np.array(4)).shape == tuple())
        self.assertTrue(discrete.information_mutual_conditional(np.array((4,)),np.array((4,)),np.array((4,))).shape == tuple())
        self.assertTrue(discrete.information_mutual_conditional(np.ones((1,)),np.ones((1,)),np.ones((1,))).shape == tuple())
        self.assertTrue(discrete.information_mutual_conditional(np.ones((1,1)),np.ones((1,1)),np.ones((1,1))).shape == (1,))        
        #Tests using cartesian_product=True
        self.assertTrue(discrete.information_mutual_conditional(np.array(4),np.array(4),np.array(4),True).shape == tuple())        
        self.assertTrue(discrete.information_mutual_conditional(np.array((4,)),np.array((4,)),np.array((4,)),True).shape == tuple())
        self.assertTrue(discrete.information_mutual_conditional(np.ones((1,)),np.ones((1,)),np.ones((1,)),True).shape == tuple())
        self.assertTrue(discrete.information_mutual_conditional(np.ones((1,1)),np.ones((1,1)),np.ones((1,1)),True).shape == (1,1,1))
        self.assertTrue(discrete.information_mutual_conditional(np.ones((1,)),np.ones((1,1)),np.ones((1,1)),True).shape == (1,1))
        self.assertTrue(discrete.information_mutual_conditional(np.ones((1,1)),np.ones((1,)),np.ones((1,1)),True).shape == (1,1))
        self.assertTrue(discrete.information_mutual_conditional(np.ones((1,1)),np.ones((1,1)),np.ones((1,)),True).shape == (1,1))
        
        with self.assertRaises(ValueError):
            discrete.information_mutual_conditional(np.array(()), np.array((1,2,3)), np.array((1,2,3)))
        with self.assertRaises(ValueError):
            discrete.information_mutual_conditional(np.array((1,2,3)), np.array(()), np.array((1,2,3)))
        with self.assertRaises(ValueError):
            discrete.information_mutual_conditional(np.array((1,2,3)), np.array((1,2,3)), np.array(()))            
        with self.assertRaises(ValueError):
            discrete.information_mutual_conditional(np.array((np.nan,1,2)), np.array((1,2,3)), np.array((1,2,3)))
        with self.assertRaises(ValueError):
            discrete.information_mutual_conditional(np.array((1,1,2)), np.array((1,2,np.nan)), np.array((1,2,3)))            
        with self.assertRaises(ValueError):
            discrete.information_mutual_conditional(np.array((1,1,2)), np.array((1,2,3)), np.array((1,2,np.nan)))
        with self.assertRaises(ValueError):
            discrete.information_mutual_conditional(np.array((1,2,3)), np.array((1,2,3)), np.array((1,2,3)), Alphabet_X=np.array((np.nan,1,2,3)), Alphabet_Y=np.array((1,2,3)), Alphabet_Z=np.array((1,2,3)))
        with self.assertRaises(ValueError):
            discrete.information_mutual_conditional(np.array((1,2,3)), np.array((1,2,3)), np.array((1,2,3)), Alphabet_X=np.array((1,2,3)), Alphabet_Y=np.array((np.nan,1,2,3)), Alphabet_Z=np.array((1,2,3)))
        with self.assertRaises(ValueError):
            discrete.information_mutual_conditional(np.array((1,2,3)), np.array((1,2,3)), np.array((1,2,3)), Alphabet_X=np.array((1,2,3)), Alphabet_Y=np.array((1,2,3)), Alphabet_Z=np.array((np.nan,1,2,3)))
        with self.assertRaises(ValueError):
            discrete.information_mutual_conditional(np.array((1,2,3)), np.array((1,2,3)), np.array((1,2,3)), fill_value=np.nan)
        with self.assertRaises(ValueError):
            discrete.information_mutual_conditional(np.array(((1,2,3),(4,5,6))), np.array((4,1,2)), np.array((4,1,2)))
        try:
            discrete.information_mutual_conditional(np.array(((1,2,3),(4,5,6))), np.array((4,1,2)), np.array((4,1,2)), True)
        except ValueError:
            self.fail("raised ValueError unexpectedly")
        with self.assertRaises(ValueError):
            discrete.information_mutual_conditional(np.array((2, 1)), np.array((2, 1)), np.array((2, 1)), base=-1)
            
        #
        # Tests using missing data
        #
        #Immutability test
        np.random.seed(4759)
        X1 = np.random.randint(16, size=(10,10)) * 1.0
        X1[0,0] = -1
        X1_copy = np.copy(X1)
        X2 = np.random.randint(16, size=(10,10))
        X2[0,1] = -1
        X2_copy = np.copy(X2)
        X3 = np.random.randint(16, size=(10,10))
        X3[0,1] = -1
        X3_copy = np.copy(X3)
        discrete.information_mutual_conditional(X1,X2,X3, fill_value=-1)
        self.assertTrue(np.all(X1_copy == X1) and np.all(X2_copy == X2) and np.all(X3_copy == X3))
        self.assertFalse(np.all(discrete.information_mutual_conditional(X1,X2,X3,fill_value=-1) == discrete.information_mutual_conditional(X1,X2,X3, fill_value=None)))
        
        # Tests using missing data involving masked arrays
        X1 = np.random.randint(16, size=(10,10)) * 1.0
        X1[0,0] = -1
        Mask1 = np.zeros_like(X1)
        Mask1[0,0] = 1
        X2 = np.random.randint(16, size=(10,10)) * 1.0
        X2[0,1] = -1
        Mask2 = np.zeros_like(X2)
        Mask2[0,1] = 1
        X3 = np.random.randint(16, size=(10,10)) * 1.0
        X3[0,2] = -1
        Mask3 = np.zeros_like(X3)
        Mask3[0,2] = 1
        self.assertTrue(np.all(discrete.information_mutual_conditional(X1, X2, X3, fill_value=-1) == discrete.information_mutual_conditional(np.ma.array(X1,mask=Mask1), X2, X3, fill_value=-1)))
        self.assertTrue(np.all(discrete.information_mutual_conditional(X1, X2, X3, fill_value=-1) == discrete.information_mutual_conditional(X1, np.ma.array(X2,mask=Mask2), X3, fill_value=-1)))
        self.assertTrue(np.all(discrete.information_mutual_conditional(X1, X2, X3, fill_value=-1) == discrete.information_mutual_conditional(X1, X2, np.ma.array(X3,mask=Mask3), fill_value=-1)))
        self.assertTrue(np.all(discrete.information_mutual_conditional(X1, X2, X3, fill_value=-1) == discrete.information_mutual_conditional(np.ma.array(X1,mask=Mask1), np.ma.array(X2,mask=Mask2), X3, fill_value=-1)))
        self.assertTrue(np.all(discrete.information_mutual_conditional(X1, X2, X3, fill_value=-1) == discrete.information_mutual_conditional(X1, np.ma.array(X2,mask=Mask2), np.ma.array(X3,mask=Mask3), fill_value=-1)))
        self.assertTrue(np.all(discrete.information_mutual_conditional(X1, X2, X3, fill_value=-1) == discrete.information_mutual_conditional(np.ma.array(X1,mask=Mask1), X2, np.ma.array(X3,mask=Mask3), fill_value=-1)))
        self.assertTrue(np.all(discrete.information_mutual_conditional(X1, X2, X3, fill_value=-1) == discrete.information_mutual_conditional(np.ma.array(X1,mask=Mask1), np.ma.array(X2,mask=Mask2), np.ma.array(X3,mask=Mask3), fill_value=-1)))        
        
        
        ### Tests involving Alphabet and non-ML estimation
        ###
        ###
        #Alphabet immutability 
        X1 = np.random.randint(16, size=(10,10))
        X2 = np.copy(X1)
        discrete.information_mutual_conditional(X2, X2, X2, estimator=1, Alphabet_X=X2, Alphabet_Y=X2, Alphabet_Z=X2)        
        self.assertTrue(np.all(X2 == X1))
        #No explicit alphabet
        self.assertTrue(discrete.information_mutual_conditional((1,2,1,2),(2,2,2,1),(2,1,1,1),estimator=1, Alphabet_X=None, Alphabet_Y=None, base=3) == discrete.entropy_joint(np.array(((1,2,1,2),(2,1,1,1))), estimator=1, base=3)+discrete.entropy_joint(np.array(((2,2,2,1),(2,1,1,1))), estimator=1, base=3)-discrete.entropy_joint(np.array(((1,2,1,2),(2,2,2,1),(2,1,1,1))), estimator=1, base=3)-discrete.entropy_joint(np.array(((2,1,1,1))), estimator=1, base=3))
        #No explicit alphabet -- Cartesian product
        A = discrete.information_mutual_conditional(np.array(((1,2,1,2),(1,2,1,2))), np.array(((2,2,2,1),(2,2,2,1))), np.array(((2,1,1,1),(2,1,1,1))), cartesian_product=True, estimator=1, Alphabet_X=None, Alphabet_Y=None, base=3)
        self.assertTrue(np.all(A == discrete.information_mutual_conditional((1,2,1,2),(2,2,2,1),(2,1,1,1),estimator=1, Alphabet_X=None, Alphabet_Y=None, base=3)))
        #Larger alphabet
        self.assertTrue(discrete.information_mutual_conditional((1,2,1,2),(2,2,2,1),(2,1,1,1),estimator=1, Alphabet_X=(1,2,3),Alphabet_Y=None,Alphabet_Z=None,base=3) == discrete.entropy_joint(np.array(((1,2,1,2),(2,1,1,1))), Alphabet_X=discrete._vstack_pad_with_fillvalue((np.array((1,2,3)),np.array((1,2))),-1), estimator=1, base=3)+discrete.entropy_joint(np.array(((2,2,2,1),(2,1,1,1))), estimator=1, base=3)-discrete.entropy_joint(np.array(((1,2,1,2),(2,2,2,1),(2,1,1,1))), Alphabet_X=discrete._vstack_pad_with_fillvalue((np.array((1,2,3)),np.array((1,2)),np.array((1,2))),-1), estimator=1, base=3)-discrete.entropy_joint(np.array(((2,1,1,1))), estimator=1, base=3))
        self.assertTrue(discrete.information_mutual_conditional((1,2,1,2),(2,2,2,1),(2,1,1,1),estimator=1, Alphabet_X=None,Alphabet_Y=(1,2,3),Alphabet_Z=None,base=3) == discrete.entropy_joint(np.array(((1,2,1,2),(2,1,1,1))), estimator=1, base=3)+discrete.entropy_joint(np.array(((2,2,2,1),(2,1,1,1))), estimator=1, base=3, Alphabet_X=discrete._vstack_pad_with_fillvalue((np.array((1,2,3)),np.array((1,2))),-1))-discrete.entropy_joint(np.array(((1,2,1,2),(2,2,2,1),(2,1,1,1))), Alphabet_X=discrete._vstack_pad_with_fillvalue((np.array((1,2)),np.array((1,2,3)),np.array((1,2))),-1), estimator=1, base=3)-discrete.entropy_joint(np.array(((2,1,1,1))), estimator=1, base=3))
        self.assertTrue(discrete.information_mutual_conditional((1,2,1,2),(2,2,2,1),(2,1,1,1),estimator=1, Alphabet_X=(1,2,3),Alphabet_Y=(1,2,3),Alphabet_Z=None,base=3) == discrete.entropy_joint(np.array(((1,2,1,2),(2,1,1,1))), Alphabet_X=discrete._vstack_pad_with_fillvalue((np.array((1,2,3)),np.array((1,2))),-1), estimator=1, base=3)+discrete.entropy_joint(np.array(((2,2,2,1),(2,1,1,1))), Alphabet_X=discrete._vstack_pad_with_fillvalue((np.array((1,2,3)),np.array((1,2))),-1), estimator=1, base=3)-discrete.entropy_joint(np.array(((1,2,1,2),(2,2,2,1),(2,1,1,1))), Alphabet_X=discrete._vstack_pad_with_fillvalue((np.array((1,2,3)),np.array((1,2,3)),np.array((1,2))),-1), estimator=1, base=3)-discrete.entropy_joint(np.array(((2,1,1,1))), estimator=1, base=3))
        self.assertTrue(discrete.information_mutual_conditional((1,2,1,2),(2,2,2,1),(2,1,1,1),estimator=1, Alphabet_X=None,Alphabet_Y=None,Alphabet_Z=(1,2,3),base=3) == discrete.entropy_joint(np.array(((1,2,1,2),(2,1,1,1))), Alphabet_X=discrete._vstack_pad_with_fillvalue((np.array((1,2)),np.array((1,2,3))),-1), estimator=1, base=3)+discrete.entropy_joint(np.array(((2,2,2,1),(2,1,1,1))), Alphabet_X=discrete._vstack_pad_with_fillvalue((np.array((1,2)),np.array((1,2,3))),-1), estimator=1, base=3)-discrete.entropy_joint(np.array(((1,2,1,2),(2,2,2,1),(2,1,1,1))), Alphabet_X=discrete._vstack_pad_with_fillvalue((np.array((1,2)),np.array((1,2)),np.array((1,2,3))),-1), estimator=1, base=3)-discrete.entropy_joint(np.array(((2,1,1,1))), estimator=1, base=3, Alphabet_X=(1,2,3)))
        self.assertTrue(discrete.information_mutual_conditional((1,2,1,2),(2,2,2,1),(2,1,1,1),estimator=1, Alphabet_X=(1,2,3),Alphabet_Y=None,Alphabet_Z=(1,2,3),base=3) == discrete.entropy_joint(np.array(((1,2,1,2),(2,1,1,1))), Alphabet_X=discrete._vstack_pad_with_fillvalue((np.array((1,2,3)),np.array((1,2,3))),-1), estimator=1, base=3)+discrete.entropy_joint(np.array(((2,2,2,1),(2,1,1,1))), Alphabet_X=discrete._vstack_pad_with_fillvalue((np.array((1,2)),np.array((1,2,3))),-1), estimator=1, base=3)-discrete.entropy_joint(np.array(((1,2,1,2),(2,2,2,1),(2,1,1,1))), Alphabet_X=discrete._vstack_pad_with_fillvalue((np.array((1,2,3)),np.array((1,2)),np.array((1,2,3))),-1), estimator=1, base=3)-discrete.entropy_joint(np.array(((2,1,1,1))), estimator=1, base=3, Alphabet_X=(1,2,3)))
        self.assertTrue(discrete.information_mutual_conditional((1,2,1,2),(2,2,2,1),(2,1,1,1),estimator=1, Alphabet_X=None,Alphabet_Y=(1,2,3),Alphabet_Z=(1,2,3),base=3) == discrete.entropy_joint(np.array(((1,2,1,2),(2,1,1,1))), Alphabet_X=discrete._vstack_pad_with_fillvalue((np.array((1,2)),np.array((1,2,3))),-1), estimator=1, base=3)+discrete.entropy_joint(np.array(((2,2,2,1),(2,1,1,1))), Alphabet_X=discrete._vstack_pad_with_fillvalue((np.array((1,2,3)),np.array((1,2,3))),-1), estimator=1, base=3)-discrete.entropy_joint(np.array(((1,2,1,2),(2,2,2,1),(2,1,1,1))), Alphabet_X=discrete._vstack_pad_with_fillvalue((np.array((1,2)),np.array((1,2,3)),np.array((1,2,3))),-1), estimator=1, base=3)-discrete.entropy_joint(np.array(((2,1,1,1))), estimator=1, base=3, Alphabet_X=(1,2,3)))
        self.assertTrue(discrete.information_mutual_conditional((1,2,1,2),(2,2,2,1),(2,1,1,1),estimator=1, Alphabet_X=(1,2,3),Alphabet_Y=(1,2,3),Alphabet_Z=(1,2,3),base=3) == discrete.entropy_joint(np.array(((1,2,1,2),(2,1,1,1))), Alphabet_X=discrete._vstack_pad_with_fillvalue((np.array((1,2,3)),np.array((1,2,3))),-1), estimator=1, base=3)+discrete.entropy_joint(np.array(((2,2,2,1),(2,1,1,1))), Alphabet_X=discrete._vstack_pad_with_fillvalue((np.array((1,2,3)),np.array((1,2,3))),-1), estimator=1, base=3)-discrete.entropy_joint(np.array(((1,2,1,2),(2,2,2,1),(2,1,1,1))), Alphabet_X=discrete._vstack_pad_with_fillvalue((np.array((1,2,3)),np.array((1,2,3)),np.array((1,2,3))),-1), estimator=1, base=3)-discrete.entropy_joint(np.array(((2,1,1,1))), estimator=1, base=3, Alphabet_X=(1,2,3)))
        #Larger alphabet -- Cartesian product
        X = np.random.randint(3, size=(10,10)) + 1
        Y = np.random.randint(3, size=(10,10)) + 1
        Z = np.random.randint(3, size=(10,10)) + 1
        A = discrete.information_mutual_conditional(X, Y, Z, cartesian_product=True, base=3, estimator=1, Alphabet_X=np.tile((1,2,3),(10,1)),Alphabet_Y=None,Alphabet_Z=None)
        B = discrete.information_mutual_conditional(X, Y, Z, cartesian_product=True, base=3, estimator=1, Alphabet_X=None,Alphabet_Y=np.tile((1,2,3),(10,1)),Alphabet_Z=None)
        C = discrete.information_mutual_conditional(X, Y, Z, cartesian_product=True, base=3, estimator=1, Alphabet_X=np.tile((1,2,3),(10,1)),Alphabet_Y=np.tile((1,2,3),(10,1)),Alphabet_Z=None)
        D = discrete.information_mutual_conditional(X, Y, Z, cartesian_product=True, base=3, estimator=1, Alphabet_X=None,Alphabet_Y=None,Alphabet_Z=np.tile((1,2,3),(10,1)))
        E = discrete.information_mutual_conditional(X, Y, Z, cartesian_product=True, base=3, estimator=1, Alphabet_X=np.tile((1,2,3),(10,1)),Alphabet_Y=None,Alphabet_Z=np.tile((1,2,3),(10,1)))
        F = discrete.information_mutual_conditional(X, Y, Z, cartesian_product=True, base=3, estimator=1, Alphabet_X=None,Alphabet_Y=np.tile((1,2,3),(10,1)),Alphabet_Z=np.tile((1,2,3),(10,1)))
        G = discrete.information_mutual_conditional(X, Y, Z, cartesian_product=True, base=3, estimator=1, Alphabet_X=np.tile((1,2,3),(10,1)),Alphabet_Y=np.tile((1,2,3),(10,1)),Alphabet_Z=np.tile((1,2,3),(10,1)))
        for i in np.arange(A.shape[0]):
            for j in np.arange(A.shape[1]):
                for k in np.arange(A.shape[2]):
                    self.assertTrue(A[i,j,k] == discrete.information_mutual_conditional(X[i], Y[j], Z[k], base=3, estimator=1, Alphabet_X=(1,2,3),Alphabet_Y=None,Alphabet_Z=None))
                    self.assertTrue(B[i,j,k] == discrete.information_mutual_conditional(X[i], Y[j], Z[k], base=3, estimator=1, Alphabet_X=None,Alphabet_Y=(1,2,3),Alphabet_Z=None))
                    self.assertTrue(C[i,j,k] == discrete.information_mutual_conditional(X[i], Y[j], Z[k], base=3, estimator=1, Alphabet_X=(1,2,3),Alphabet_Y=(1,2,3),Alphabet_Z=None))                    
                    self.assertTrue(D[i,j,k] == discrete.information_mutual_conditional(X[i], Y[j], Z[k], base=3, estimator=1, Alphabet_X=None,Alphabet_Y=None,Alphabet_Z=(1,2,3)))
                    self.assertTrue(E[i,j,k] == discrete.information_mutual_conditional(X[i], Y[j], Z[k], base=3, estimator=1, Alphabet_X=(1,2,3),Alphabet_Y=None,Alphabet_Z=(1,2,3)))
                    self.assertTrue(F[i,j,k] == discrete.information_mutual_conditional(X[i], Y[j], Z[k], base=3, estimator=1, Alphabet_X=None,Alphabet_Y=(1,2,3),Alphabet_Z=(1,2,3)))
                    self.assertTrue(G[i,j,k] == discrete.information_mutual_conditional(X[i], Y[j], Z[k], base=3, estimator=1, Alphabet_X=(1,2,3),Alphabet_Y=(1,2,3),Alphabet_Z=(1,2,3)))
        #Empty alphabet
        with self.assertRaises(ValueError):
            discrete.information_mutual_conditional([1, 2], [1, 2], [1, 2], estimator=1, Alphabet_X=np.array(()), Alphabet_Y=None, Alphabet_Z=None, base=3)
        with self.assertRaises(ValueError):
            discrete.information_mutual_conditional([1, 2], [1, 2], [1, 2], estimator=1, Alphabet_X=None, Alphabet_Y=np.array(()), Alphabet_Z=None, base=3)
        with self.assertRaises(ValueError):
            discrete.information_mutual_conditional([1, 2], [1, 2], [1, 2], estimator=1, Alphabet_X=None, Alphabet_Y=None, Alphabet_Z=np.array(()), base=3)            
        #Smaller alphabet
        with self.assertRaises(ValueError):
            discrete.information_mutual_conditional([1,2], [1,2], [1,2], estimator=1, Alphabet_X=[1,1], Alphabet_Y=[1,2], Alphabet_Z=[1,2], base=3)
        with self.assertRaises(ValueError):
            discrete.information_mutual_conditional([1,2], [1,2], [1,2], estimator=1, Alphabet_X=[1,2], Alphabet_Y=[1,1], Alphabet_Z=[1,2], base=3)
        with self.assertRaises(ValueError):
            discrete.information_mutual_conditional([1,2], [1,2], [1,2], estimator=1, Alphabet_X=[1,2], Alphabet_Y=[1,2], Alphabet_Z=[1,1], base=3)
        with self.assertRaises(ValueError):
            discrete.information_mutual_conditional(np.array(((1,2),(1,2))), np.array(((1,2),(1,2))), np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array(((1,2),(1,None))), Alphabet_Y=np.array(((1,2),(1,2))), Alphabet_Z=np.array(((1,2),(1,2))),base=3)
        with self.assertRaises(ValueError):
            discrete.information_mutual_conditional(np.array(((1,2),(1,2))), np.array(((1,2),(1,2))), np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array(((1,2),(1,2))), Alphabet_Y=np.array(((1,2),(1,None))), Alphabet_Z=np.array(((1,2),(1,2))),base=3)
        with self.assertRaises(ValueError):
            discrete.information_mutual_conditional(np.array(((1,2),(1,2))), np.array(((1,2),(1,2))), np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array(((1,2),(1,2))), Alphabet_Y=np.array(((1,2),(1,2))), Alphabet_Z=np.array(((1,2),(1,None))),base=3)            
        #Alphabet with incorrect dimensions
        with self.assertRaises(ValueError):
            discrete.information_mutual_conditional(np.array(((1,2),(1,2))), np.array(((1,2),(1,2))), np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array(((1,2))), Alphabet_Y=np.array(((1,2),(1,2))), Alphabet_Z=np.array(((1,2),(1,2))), base=3)
        with self.assertRaises(ValueError):
            discrete.information_mutual_conditional(np.array(((1,2),(1,2))), np.array(((1,2),(1,2))), np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array(((1,2),(1,2))), Alphabet_Y=np.array(((1,2))), Alphabet_Z=np.array(((1,2),(1,2))), base=3)
        with self.assertRaises(ValueError):
            discrete.information_mutual_conditional(np.array(((1,2),(1,2))), np.array(((1,2),(1,2))), np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array(((1,2),(1,2))), Alphabet_Y=np.array(((1,2),(1,2))), Alphabet_Z=np.array(((1,2))), base=3)            
            
    def test_information_co(self):
        #Immutability test
        X1 = np.random.randint(16, size=(10,10))
        X1_copy = np.copy(X1)
        discrete.information_co(X1)
        self.assertTrue(np.all(X1_copy == X1))
                
        self.assertTrue(discrete.information_co(np.array((1.0, 1.0, 1.0, 1.0)), base=2) == 0)
        self.assertTrue(discrete.information_co(np.array((1.0, 1.0, 2.0, 2.0)), base=2) == 1)
        X = np.array((1,1,2,2))
        Y = np.array((2,2,2,2))
        self.assertTrue(discrete.information_co(np.vstack((X,Y)), base=2) == 0)
        Y = np.array((1,1,2,2))
        self.assertTrue(discrete.information_co(np.vstack((X,Y)), base=2) == 1)
        X = np.array((2,1,2,2))
        Y = np.array((1,1,2,2))
        self.assertTrue(np.allclose(discrete.information_co(np.vstack((X,Y)), base=np.exp(1)), discrete.information_mutual(X,Y, base=np.exp(1))))
        
        self.assertTrue(discrete.information_co(np.ones(5)).shape == tuple())
        self.assertTrue(discrete.information_co(np.ones((5,1)).T).shape == tuple())
        self.assertTrue(discrete.information_co(np.ones((5,1))).shape == tuple())
        self.assertTrue(discrete.information_co(np.ones((5,1))).shape == tuple())
        
        self.assertTrue(discrete.information_co(np.array(4)).shape == tuple())
        self.assertTrue(discrete.information_co(np.array((4,))).shape == tuple())
        self.assertTrue(discrete.information_co(np.ones((1,))).shape == tuple())
        self.assertTrue(discrete.information_co(np.ones((1,1))).shape == tuple())        
            
        with self.assertRaises(ValueError):
            discrete.information_co(np.array(()))
        with self.assertRaises(ValueError):
            discrete.information_co(np.array((np.nan,1,2)))
        with self.assertRaises(ValueError):
            discrete.information_co(np.array((1,2)), Alphabet_X=np.array((1,2,np.nan)))
        with self.assertRaises(ValueError):
            discrete.information_co(np.array((2,1)), fill_value=np.nan)
        with self.assertRaises(ValueError):
            discrete.information_co(np.array((2,1)), base=-1)            
            
        #
        # Tests using missing data (basic)
        #
        #Immutability test        
        X1 = np.random.randint(16, size=(10,10)) * 1.0
        X1[0,0] = -1
        X1_copy = np.copy(X1)
        discrete.information_co(X1, fill_value=-1)
        self.assertTrue(np.all(X1_copy == X1))
        self.assertFalse(np.all(discrete.information_co(X1,fill_value=-1) == discrete.information_co(X1, fill_value=None)))
        
        # Tests using missing data involving masked arrays
        X1 = np.random.randint(16, size=(10,10)) * 1.0
        X1[0,0] = -1
        Mask = np.zeros_like(X1)
        Mask[0,0] = 1
        self.assertTrue(discrete.information_co(X1, fill_value=-1) == discrete.information_co(np.ma.array(X1,mask=Mask)))        
        
        # Tests using missing data and None values
        self.assertTrue(discrete.information_co((1,2,1,2,1,1,None,None), fill_value=-1) == discrete.information_co((1,2,1,2,1,1,3,3)))
        self.assertTrue(discrete.information_co((1,2,1,2,1,1,None,None), fill_value=None) == discrete.information_co((1,2,1,2,1,1)))
        self.assertTrue(discrete.information_co(np.ma.array((1,2,1,2,1,1,None,None),mask=(0,0,0,0,0,0,1,1))) == discrete.information_co((1,2,1,2,1,1)))
        self.assertTrue(discrete.information_co((1,2,1,2,1,1), Alphabet_X=(1,2,None), fill_value=-1) == discrete.information_co((1,2,1,2,1,1), Alphabet_X=(1,2,3)))
        self.assertTrue(discrete.information_co((1,2,1,2,1,1), Alphabet_X=(1,2,None), fill_value=None) == discrete.information_co((1,2,1,2,1,1), Alphabet_X=(1,2)))
        self.assertTrue(discrete.information_co((1,2,1,2,1,1), Alphabet_X=np.ma.array((1,2,None),mask=(0,0,1))) == discrete.information_co((1,2,1,2,1,1), Alphabet_X=(1,2)))        
        self.assertTrue(discrete.information_co((1,2,1,2,1,1,None,None), Alphabet_X=(1,2,None), fill_value=-1) == discrete.information_co((1,2,1,2,1,1,3,3), Alphabet_X=(1,2,3)))
        self.assertTrue(discrete.information_co((1,2,1,2,1,1,None,None), Alphabet_X=(1,2,None), fill_value=None) == discrete.information_co((1,2,1,2,1,1), Alphabet_X=(1,2)))
        self.assertTrue(discrete.information_co(np.ma.array((1,2,1,2,1,1,None,None),mask=(0,0,0,0,0,0,1,1)), Alphabet_X=np.ma.array((1,2,3),mask=(0,0,1))) == discrete.information_co((1,2,1,2,1,1), Alphabet_X=(1,2)))        
        
        ### Tests involving Alphabet and non-ML estimation
        ###
        ###
        #Alphabet immutability 
        X1 = np.random.randint(16, size=(10,10))
        X2 = np.copy(X1)
        discrete.information_co(X2, estimator=1, Alphabet_X=X2)
        self.assertTrue(np.all(X2 == X1))
        #No explicit alphabet
        self.assertTrue(discrete.information_co(np.array(((1,2,1),(1,2,2))),estimator=1,Alphabet_X=None,base=3) == -discrete.entropy_joint(np.array(((1,2,1),(1,2,2))), estimator=1,Alphabet_X=None,base=3)+2*discrete.entropy([1,2,1],estimator=1,base=3))
        #Larger alphabet
        self.assertTrue(discrete.information_co(np.array(((1,2,1),(1,2,2))),estimator=1,Alphabet_X=np.array(((1,2,3),(1,2,3))),base=3) == -discrete.entropy_joint(np.array(((1,2,1),(1,2,2))), estimator=1,Alphabet_X=np.array(((1,2,3),(1,2,3))),base=3)+2*discrete.entropy([1,2,1],estimator=1,base=3,Alphabet_X=np.array((1,2,3))))
        #Empty alphabet
        with self.assertRaises(ValueError):
            discrete.information_co([1, 2], estimator=1, Alphabet_X=np.array(()), base=3)
        #Smaller alphabet
        with self.assertRaises(ValueError):
            discrete.information_co([1, 2], estimator=1, Alphabet_X=[1,], base=3)
        #Alphabet with incorrect dimensions
        with self.assertRaises(ValueError):
            discrete.information_co(np.array((1,2)), estimator=1, Alphabet_X=np.array(((1,2),(1,2))), base=3)        
            
    def test_information_interaction(self):
        #Immutability test
        X1 = np.random.randint(16, size=(10,10))
        X1_copy = np.copy(X1)
        discrete.information_interaction(X1)
        self.assertTrue(np.all(X1_copy == X1))
        
        self.assertTrue(discrete.information_interaction(X1) == discrete.information_co(X1))
        X1 = np.random.randint(16, size=(11,10))
        self.assertTrue(-1*discrete.information_interaction(X1) == discrete.information_co(X1))        
                
        self.assertTrue(discrete.information_interaction(np.array((1.0, 1.0, 1.0, 1.0)), base=2) == 0)
        self.assertTrue(discrete.information_interaction(np.array((1.0, 1.0, 2.0, 2.0)), base=2) == -1)
        X = np.array((1,1,2,2))
        Y = np.array((2,2,2,2))
        self.assertTrue(discrete.information_interaction(np.vstack((X,Y)), base=2) == 0)
        Y = np.array((1,1,2,2))
        self.assertTrue(discrete.information_interaction(np.vstack((X,Y)), base=2) == 1)
        X = np.array((2,1,2,2))
        Y = np.array((1,1,2,2))
        self.assertTrue(np.allclose(discrete.information_interaction(np.vstack((X,Y)), base=np.exp(1)), discrete.information_mutual(X,Y, base=np.exp(1))))
        
        self.assertTrue(discrete.information_interaction(np.ones(5)).shape == tuple())
        self.assertTrue(discrete.information_interaction(np.ones((5,1)).T).shape == tuple())
        self.assertTrue(discrete.information_interaction(np.ones((5,1))).shape == tuple())
        self.assertTrue(discrete.information_interaction(np.ones((5,1))).shape == tuple())
        
        self.assertTrue(discrete.information_interaction(np.array(4)).shape == tuple())
        self.assertTrue(discrete.information_interaction(np.array((4,))).shape == tuple())
        self.assertTrue(discrete.information_interaction(np.ones((1,))).shape == tuple())
        self.assertTrue(discrete.information_interaction(np.ones((1,1))).shape == tuple())
        
        with self.assertRaises(ValueError):
            discrete.information_interaction(np.array(()))
        with self.assertRaises(ValueError):
            discrete.information_interaction(np.array((np.nan,1,2)))
        with self.assertRaises(ValueError):
            discrete.information_interaction(np.array((1,2)), Alphabet_X=np.array((1,2,np.nan)))
        with self.assertRaises(ValueError):
            discrete.information_interaction(np.array((2,1)), fill_value=np.nan)            
        with self.assertRaises(ValueError):
            discrete.information_interaction(np.array((2,1)), base=-1)
            
        #
        # Tests using missing data (basic)
        #
        #Immutability test        
        X1 = np.random.randint(16, size=(10,10)) * 1.0
        X1[0,0] = -1
        X1_copy = np.copy(X1)
        discrete.information_interaction(X1, fill_value=-1)
        self.assertTrue(np.all(X1_copy == X1))
        self.assertFalse(np.all(discrete.information_interaction(X1,fill_value=-1) == discrete.information_interaction(X1, fill_value=None)))
        
        # Tests using missing data involving masked arrays
        X1 = np.random.randint(16, size=(10,10)) * 1.0
        X1[0,0] = -1
        Mask = np.zeros_like(X1)
        Mask[0,0] = 1
        self.assertTrue(discrete.information_interaction(X1, fill_value=-1) == discrete.information_interaction(np.ma.array(X1,mask=Mask)))        
        
        # Tests using missing data and None values
        self.assertTrue(discrete.information_interaction((1,2,1,2,1,1,None,None), fill_value=-1) == discrete.information_interaction((1,2,1,2,1,1,3,3)))
        self.assertTrue(discrete.information_interaction((1,2,1,2,1,1,None,None), fill_value=None) == discrete.information_interaction((1,2,1,2,1,1)))
        self.assertTrue(discrete.information_interaction(np.ma.array((1,2,1,2,1,1,None,None),mask=(0,0,0,0,0,0,1,1))) == discrete.information_interaction((1,2,1,2,1,1)))
        self.assertTrue(discrete.information_interaction((1,2,1,2,1,1), Alphabet_X=(1,2,None), fill_value=-1) == discrete.information_interaction((1,2,1,2,1,1), Alphabet_X=(1,2,3)))
        self.assertTrue(discrete.information_interaction((1,2,1,2,1,1), Alphabet_X=(1,2,None), fill_value=None) == discrete.information_interaction((1,2,1,2,1,1), Alphabet_X=(1,2)))
        self.assertTrue(discrete.information_interaction((1,2,1,2,1,1), Alphabet_X=np.ma.array((1,2,None),mask=(0,0,1))) == discrete.information_interaction((1,2,1,2,1,1), Alphabet_X=(1,2)))        
        self.assertTrue(discrete.information_interaction((1,2,1,2,1,1,None,None), Alphabet_X=(1,2,None), fill_value=-1) == discrete.information_interaction((1,2,1,2,1,1,3,3), Alphabet_X=(1,2,3)))
        self.assertTrue(discrete.information_interaction((1,2,1,2,1,1,None,None), Alphabet_X=(1,2,None), fill_value=None) == discrete.information_interaction((1,2,1,2,1,1), Alphabet_X=(1,2)))
        self.assertTrue(discrete.information_interaction(np.ma.array((1,2,1,2,1,1,None,None),mask=(0,0,0,0,0,0,1,1)), Alphabet_X=np.ma.array((1,2,3),mask=(0,0,1))) == discrete.information_interaction((1,2,1,2,1,1), Alphabet_X=(1,2)))        
        
        ### Tests involving Alphabet and non-ML estimation
        ###
        ###
        #Alphabet immutability
        X1 = np.random.randint(16, size=(10,10))
        X2 = np.copy(X1)
        discrete.information_interaction(X2, estimator=1, Alphabet_X=X2)
        self.assertTrue(np.all(X2 == X1))
        #No explicit alphabet
        self.assertTrue(discrete.information_interaction(np.array(((1,2,1),(1,2,2))),estimator=1,Alphabet_X=None,base=3) == -discrete.entropy_joint(np.array(((1,2,1),(1,2,2))), estimator=1,Alphabet_X=None,base=3)+2*discrete.entropy([1,2,1],estimator=1,base=3))
        #Larger alphabet
        self.assertTrue(discrete.information_interaction(np.array(((1,2,1),(1,2,2))),estimator=1,Alphabet_X=np.array(((1,2,3),(1,2,3))),base=3) == -discrete.entropy_joint(np.array(((1,2,1),(1,2,2))), estimator=1,Alphabet_X=np.array(((1,2,3),(1,2,3))),base=3)+2*discrete.entropy([1,2,1],estimator=1,base=3,Alphabet_X=np.array((1,2,3))))
        #Empty alphabet
        with self.assertRaises(ValueError):
            discrete.information_interaction([1, 2], estimator=1, Alphabet_X=np.array(()), base=3)
        #Smaller alphabet
        with self.assertRaises(ValueError):
            discrete.information_interaction([1, 2], estimator=1, Alphabet_X=[1,], base=3)
        #Alphabet with incorrect dimensions
        with self.assertRaises(ValueError):
            discrete.information_interaction(np.array((1,2)), estimator=1, Alphabet_X=np.array(((1,2),(1,2))), base=3)
            
    def test_information_enigmatic(self):
        ### Tests involving Alphabet and non-ML estimation
        ###
        ###
        #Alphabet immutability
        X1 = np.random.randint(16, size=(10,10))
        X2 = np.copy(X1)
        discrete.information_enigmatic(X2, estimator=1, Alphabet_X=X2)
        self.assertTrue(np.all(X2 == X1))
        #No explicit alphabet
        self.assertTrue(discrete.information_enigmatic(np.array(((1,2,1),(1,2,2))),estimator=1,Alphabet_X=None,base=3) == discrete.information_multi(np.array(((1,2,1),(1,2,2))), estimator=1,Alphabet_X=None,base=3) - discrete.information_binding(np.array(((1,2,1),(1,2,2))), estimator=1,Alphabet_X=None,base=3))
        np.array(((1,2,3),(1,2,3)))
        #Larger alphabet
        self.assertTrue(discrete.information_enigmatic(np.array(((1,2,1),(1,2,2))),estimator=1,Alphabet_X=np.array(((1,2,3),(1,2,3))),base=3) == discrete.information_multi(np.array(((1,2,1),(1,2,2))), estimator=1,Alphabet_X=np.array(((1,2,3),(1,2,3))),base=3) - discrete.information_binding(np.array(((1,2,1),(1,2,2))), estimator=1,Alphabet_X=np.array(((1,2,3),(1,2,3))),base=3))
        #Empty alphabet
        with self.assertRaises(ValueError):
            discrete.information_enigmatic([1, 2], estimator=1, Alphabet_X=np.array(()), base=3)
        #Smaller alphabet
        with self.assertRaises(ValueError):
            discrete.information_enigmatic([1, 2], estimator=1, Alphabet_X=[1,], base=3)
        #Alphabet with incorrect dimensions
        with self.assertRaises(ValueError):
            discrete.information_enigmatic(np.array((1,2)), estimator=1, Alphabet_X=np.array(((1,2),(1,2))), base=3)
            
    def test_information_exogenous_local(self):
        ### Tests involving Alphabet and non-ML estimation
        ###
        ###
        #Alphabet immutability
        X1 = np.random.randint(16, size=(10,10))
        X2 = np.copy(X1)
        discrete.information_exogenous_local(X2, estimator=1, Alphabet_X=X2)
        self.assertTrue(np.all(X2 == X1))
        #No explicit alphabet
        self.assertTrue(discrete.information_exogenous_local(np.array(((1,2,1),(1,2,2))),estimator=1,Alphabet_X=None,base=3) == discrete.information_binding(np.array(((1,2,1),(1,2,2))), estimator=1,Alphabet_X=None,base=3) + discrete.information_multi(np.array(((1,2,1),(1,2,2))), estimator=1,Alphabet_X=None,base=3))
        np.array(((1,2,3),(1,2,3)))
        #Larger alphabet
        self.assertTrue(discrete.information_exogenous_local(np.array(((1,2,1),(1,2,2))),estimator=1,Alphabet_X=np.array(((1,2,3),(1,2,3))),base=3) == discrete.information_binding(np.array(((1,2,1),(1,2,2))), estimator=1,Alphabet_X=np.array(((1,2,3),(1,2,3))),base=3) + discrete.information_multi(np.array(((1,2,1),(1,2,2))), estimator=1,Alphabet_X=np.array(((1,2,3),(1,2,3))),base=3))
        #Empty alphabet
        with self.assertRaises(ValueError):
            discrete.information_exogenous_local([1, 2], estimator=1, Alphabet_X=np.array(()), base=3)
        #Smaller alphabet
        with self.assertRaises(ValueError):
            discrete.information_exogenous_local([1, 2], estimator=1, Alphabet_X=[1,], base=3)
        #Alphabet with incorrect dimensions
        with self.assertRaises(ValueError):
            discrete.information_exogenous_local(np.array((1,2)), estimator=1, Alphabet_X=np.array(((1,2),(1,2))), base=3)
            
    def test_entropy_residual(self):
        ### Tests involving Alphabet and non-ML estimation
        ###
        ###
        #Alphabet immutability
        X1 = np.random.randint(16, size=(10,10))
        X2 = np.copy(X1)
        discrete.entropy_residual(X2, estimator=1, Alphabet_X=X2)
        self.assertTrue(np.all(X2 == X1))
        #No explicit alphabet
        self.assertTrue(discrete.entropy_residual(np.array(((1,2,1),(1,2,2))),estimator=1,Alphabet_X=None,base=3) == discrete.entropy_joint(np.array(((1,2,1),(1,2,2))), estimator=1,Alphabet_X=None,base=3) - discrete.information_binding(np.array(((1,2,1),(1,2,2))), estimator=1,Alphabet_X=None,base=3))
        np.array(((1,2,3),(1,2,3)))
        #Larger alphabet
        self.assertTrue(discrete.entropy_residual(np.array(((1,2,1),(1,2,2))),estimator=1,Alphabet_X=np.array(((1,2,3),(1,2,3))),base=3) == discrete.entropy_joint(np.array(((1,2,1),(1,2,2))), estimator=1,Alphabet_X=np.array(((1,2,3),(1,2,3))),base=3) - discrete.information_binding(np.array(((1,2,1),(1,2,2))), estimator=1,Alphabet_X=np.array(((1,2,3),(1,2,3))),base=3))
        #Empty alphabet
        with self.assertRaises(ValueError):
            discrete.entropy_residual([1, 2], estimator=1, Alphabet_X=np.array(()), base=3)
        #Smaller alphabet
        with self.assertRaises(ValueError):
            discrete.entropy_residual([1, 2], estimator=1, Alphabet_X=[1,], base=3)
        #Alphabet with incorrect dimensions
        with self.assertRaises(ValueError):
            discrete.entropy_residual(np.array((1,2)), estimator=1, Alphabet_X=np.array(((1,2),(1,2))), base=3)            
        
    def test_information_multi(self):
        ### Tests involving Alphabet and non-ML estimation
        ###
        ###
        #Alphabet immutability 
        X1 = np.random.randint(16, size=(10,10))
        X2 = np.copy(X1)
        discrete.information_multi(X2, estimator=1, Alphabet_X=X2)
        self.assertTrue(np.all(X2 == X1))
        #No explicit alphabet
        self.assertTrue(discrete.information_multi(np.array(((1,2),(1,2))), estimator=1, Alphabet_X=None) == 2-discrete.entropy_joint(np.array(((1,1,1,2,2,2),(1,1,2,1,2,2))), Alphabet_X=None))
        #Larger alphabet
        self.assertTrue(np.abs(discrete.information_multi(np.array(((1,2),(1,2))), estimator=1, Alphabet_X=np.array(([1,2,3],[1,2,3])), base=3)- (2*discrete.entropy([1,2],Alphabet_X=[1,2,3],estimator=1,base=3)-discrete.entropy_joint(np.array(((1,1,1,1,2,2,2,2,3,3,3),(1,1,2,3,1,2,2,3,1,2,3))), Alphabet_X=None, base=3)))< 1E-03)
        #Empty alphabet
        with self.assertRaises(ValueError):
            discrete.information_multi([1, 2], estimator=1, Alphabet_X=np.array(()), base=3)
        #Smaller alphabet
        with self.assertRaises(ValueError):
            discrete.information_multi([1, 2], estimator=1, Alphabet_X=[1,], base=3)
        #Alphabet with incorrect dimensions
        with self.assertRaises(ValueError):
            discrete.information_multi(np.array((1,2)), estimator=1, Alphabet_X=np.array(((1,2),(1,2))), base=3)        
        
    def test__estimate_probabilities(self):  
        
        #James-Stein estimator
        self.assertTrue(discrete._estimate_probabilities(np.array((1,)), 'james-stein') == (1,0))
        self.assertTrue(np.allclose(np.sum(discrete._estimate_probabilities(np.random.randint(0,1000,1000), 'james-stein')[0]), 1))
        for n in range(10):
            P,P_0 = discrete._estimate_probabilities(np.array((6,4)), 'james-stein',n)
            self.assertTrue(np.allclose(np.sum(P) + n*P_0, 1))
            Q,Q_0 = discrete._estimate_probabilities(np.append(np.array((6,4)), np.zeros(n)), 'james-stein')
            self.assertTrue(np.allclose(Q, np.append(P, np.tile(P_0,n))))
        for n in range(10):
            P,P_0 = discrete._estimate_probabilities(np.array((6,0,4)), 'james-stein',n)
            self.assertTrue(np.allclose(np.sum(P) + n*P_0, 1))
        #Some hand-worked examples
        #[1 2 8]; 0 empty bins
        Theta = np.array((1,2,8)) / 11.0
        t_k = 1 / 3.0
        Lambda = (1 - np.sum(Theta**2)) / ((11-1) * np.sum((t_k - Theta)**2))
        Theta_shrink = Lambda*t_k + (1-Lambda)*Theta
        self.assertTrue(np.all(Theta_shrink == discrete._estimate_probabilities(np.array((1,2,8)), 'james-stein',0)[0]))
        #0.1 0.2 0.8; 1 empty bin
        Theta = np.array((1,2,8)) / 11.0
        t_k = 1 / 4.0
        Lambda = (1 - np.sum(Theta**2)) / ((11-1) * (np.sum((t_k - Theta)**2)+(t_k**2)))
        Theta_shrink = Lambda*t_k + (1-Lambda)*Theta
        self.assertTrue(np.all(Theta_shrink == discrete._estimate_probabilities(np.array((1,2,8)), 'james-stein',1)[0]))        
        #0.1 0.2 0.8; 2 empty bins
        Theta = np.array((1,2,8)) / 11.0
        t_k = 1 / 5.0
        Lambda = (1 - np.sum(Theta**2)) / ((11-1) * (np.sum((t_k - Theta)**2)+(2*t_k**2)))
        Theta_shrink = Lambda*t_k + (1-Lambda)*Theta
        self.assertTrue(np.all(Theta_shrink == discrete._estimate_probabilities(np.array((1,2,8)), 'james-stein',2)[0]))
        
        #ML/MAP estimator
        for param in ('ML', 0, 'PERKS', 'MINIMAX'):
            self.assertTrue(discrete._estimate_probabilities(np.array((1,)), param) == (1,0))
            self.assertTrue(np.allclose(np.sum(discrete._estimate_probabilities(np.random.randint(0,1000,1000), param)[0]), 1))
            for n in range(10):
                P,P_0 = discrete._estimate_probabilities(np.array((6,4)), param,n)
                self.assertTrue(np.allclose(np.sum(P) + n*P_0, 1))
                Q,Q_0 = discrete._estimate_probabilities(np.append(np.array((6,4)), np.zeros(n)), param)
                self.assertTrue(np.allclose(Q, np.append(P, np.tile(P_0,n))))        
            for n in range(10):
                P,P_0 = discrete._estimate_probabilities(np.array((6,0,4)), param,n)
                self.assertTrue(np.allclose(np.sum(P) + n*P_0, 1))
        #Some hand-worked examples
        #[1 2 8]; 0 empty bins
        Theta = np.array((1,2,8)) / 11.0
        self.assertTrue(np.all(Theta == discrete._estimate_probabilities(np.array((1,2,8)), 'ML',0)[0]))
        Theta = (1.0/3+np.array((1,2,8))) / (11.0 + 1.0)
        self.assertTrue(np.all(Theta == discrete._estimate_probabilities(np.array((1,2,8)), 'PERKS',0)[0]))
        Theta = (np.sqrt(11)/3+np.array((1,2,8))) / (11.0 + np.sqrt(11))        
        self.assertTrue(np.all(Theta == discrete._estimate_probabilities(np.array((1,2,8)), 'MINIMAX',0)[0]))       
        #0.1 0.2 0.8; 1 empty bin
        Theta = np.array((1,2,8)) / 11.0
        self.assertTrue(np.all(Theta == discrete._estimate_probabilities(np.array((1,2,8)), 'ML',1)[0]))
        Theta = (1.0/4+np.array((1,2,8))) / (11.0 + 1.0)
        self.assertTrue(np.all(Theta == discrete._estimate_probabilities(np.array((1,2,8)), 'PERKS',1)[0]))
        Theta = (np.sqrt(11)/4+np.array((1,2,8))) / (11.0 + np.sqrt(11))        
        self.assertTrue(np.all(Theta == discrete._estimate_probabilities(np.array((1,2,8)), 'MINIMAX',1)[0]))
        #0.1 0.2 0.8; 2 empty bins
        Theta = np.array((1,2,8)) / 11.0
        self.assertTrue(np.all(Theta == discrete._estimate_probabilities(np.array((1,2,8)), 'ML',2)[0]))
        Theta = (1.0/5+np.array((1,2,8))) / (11.0 + 1.0)
        self.assertTrue(np.all(Theta == discrete._estimate_probabilities(np.array((1,2,8)), 'PERKS',2)[0]))
        Theta = (np.sqrt(11)/5+np.array((1,2,8))) / (11.0 + np.sqrt(11))        
        self.assertTrue(np.all(Theta == discrete._estimate_probabilities(np.array((1,2,8)), 'MINIMAX',2)[0]))       
        
        #Good-Turing estimator
        self.assertTrue(discrete._estimate_probabilities(np.array((1,)), 'good-turing') == (1,0))
        self.assertTrue(np.allclose(np.sum(discrete._estimate_probabilities(np.random.randint(0,1000,1000), 'good-turing')[0]), 1))
        for n in range(10):
            P,P_0 = discrete._estimate_probabilities(np.array((6,4)), 'good-turing',n)
            self.assertTrue(np.allclose(np.sum(P) + n*P_0, 1))
            Q,Q_0 = discrete._estimate_probabilities(np.append(np.array((6,4)), np.zeros(n)), 'good-turing')
            self.assertTrue(np.allclose(Q, np.append(P, np.tile(P_0,n))))
        for n in range(10):
            P,P_0 = discrete._estimate_probabilities(np.array((6,0,4)), 'good-turing',n)
            self.assertTrue(np.allclose(np.sum(P) + n*P_0, 1))
        #Prosody dataset given in Gale and Sampson (1995)
        N_r = np.array(((1,120),\
        (2,40),\
        (3,24),\
        (4,13),\
        (5,15),\
        (6,5),\
        (7,11),\
        (8,2),\
        (9,2),\
        (10,1),\
        (12,3),\
        (14,2),\
        (15,1),\
        (16,1),\
        (17,3),\
        (19,1),\
        (20,3),\
        (21,2),\
        (23,3),\
        (24,3),\
        (25,3),\
        (26,2),\
        (27,2),\
        (28,1),\
        (31,2),\
        (32,2),\
        (33,1),\
        (34,2),\
        (36,2),\
        (41,3),\
        (43,1),\
        (45,3),\
        (46,1),\
        (47,1),\
        (50,1),\
        (71,1),\
        (84,1),\
        (101,1),\
        (105,1),\
        (121,1),\
        (124,1),\
        (146,1),\
        (162,1),\
        (193,1),\
        (199,1),\
        (224,1),\
        (226,1),\
        (254,1),\
        (257,1),\
        (339,1),\
        (421,1),\
        (456,1),\
        (481,1),\
        (483,1),\
        (1140,1),\
        (1256,1),\
        (1322,1),\
        (1530,1),\
        (2131,1),\
        (2395,1),\
        (6925,1),\
        (7846,1)))
        N_r = N_r.tolist()
        Counts = []
        for r, n_r in N_r:
            for i in xrange(n_r):
                Counts.append(r)
        Counts = np.array(Counts)
        P = discrete._estimate_probabilities(Counts, 'good-turing',1)[0]
        self.assertTrue(np.all(np.abs(P[Counts == 7846]-0.2537)<0.001))
        self.assertTrue(np.all(np.abs(P[Counts == 6925]-0.2239)<0.001))
        self.assertTrue(np.all(np.abs(P[Counts == 224]-0.007230)<0.001))
        self.assertTrue(np.all(np.abs(P[Counts == 23]-0.0007314)<0.001))
        self.assertTrue(np.all(np.abs(P[Counts == 7]-0.0002149)<0.001))
        self.assertTrue(np.all(np.abs(P[Counts == 6]-0.0001827)<0.001))
        self.assertTrue(np.all(np.abs(P[Counts == 5]-0.0001506)<0.001))
        self.assertTrue(np.all(np.abs(P[Counts == 4]-0.0001186)<0.001))
        self.assertTrue(np.all(np.abs(P[Counts == 3]-8.672e-05)<0.001))
        self.assertTrue(np.all(np.abs(P[Counts == 2]-5.522e-05)<0.001))
        self.assertTrue(np.all(np.abs(P[Counts == 1]-2.468e-05)<0.001))        
             
if __name__ == '__main__':
    unittest.main()