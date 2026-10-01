import unittest
import numpy as np
from model import generate,Problem,solve

class Tests(unittest.TestCase):
    def setUp(self):self.p=Problem(generate())
    def test_empty(self):self.assertEqual(self.p.objective(np.zeros(50,dtype=bool)),0)
    def test_over_budget_rejected(self):
        with self.assertRaises(ValueError):self.p.objective(np.ones(50,dtype=bool))
    def test_repair(self):
        rng=np.random.default_rng(5)
        for x in rng.random((100,50))<.5:self.assertLessEqual(self.p.metrics(self.p.repair(x))[1],70)
    def test_manual_coverage(self):
        p=Problem(dict(coverage=[[1,0],[1,1]],duration=[2,3],weights=[4,5],time_limit=10))
        self.assertEqual(p.metrics(np.array([True,True])),(9,5));self.assertAlmostEqual(p.objective(np.array([True,True])),8.95)
    def test_generator_covers_all_requirements(self):self.assertTrue(self.p.matrix.any(axis=0).all())
    def test_greedy_feasible(self):self.assertLessEqual(self.p.metrics(self.p.greedy())[1],70)
    def test_budget_local_search_determinism(self):
        c=dict(population=12,budget=103,mutation_probability=.02,crossover_probability=.9,local_probability=1.)
        for mode in [False,True]:
            a=solve(c,self.p,17,mode);b=solve(c,self.p,17,mode)
            self.assertEqual(a[3],103);self.assertEqual(len(a[2]),9);np.testing.assert_array_equal(a[1],b[1]);self.assertTrue(np.all(np.diff(a[2])>=0));self.assertAlmostEqual(a[0],self.p.objective(a[1]));self.assertLessEqual(self.p.metrics(a[1])[1],70)
            if mode:self.assertGreater(a[4],0)
if __name__=='__main__':unittest.main()
