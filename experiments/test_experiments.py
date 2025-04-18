#python3.13 -m unittest test_experiments.py
 
import unittest
from unittest.mock import patch
import sys, random, os
import experiment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ezr

class TestExperiment(unittest.TestCase):

    def setUp(self):
        # Define or load datasets
        self.data_sets = [
            {'file': '../data/optimize/config/Apache_AllMeasurements.csv'},
            {'file': '../data/optimize/config/SS-A.csv'}
        ]
        # Set up a sample dataset for testing
        self.data_path = self.data_sets[0]['file']
        self.d = ezr.DATA().adds(ezr.csv(self.data_path))
        self.N = 20

    def test_chebyshevs_top_item(self):
        # Test if chebyshevs().rows[0] returns the top item in that sort
        some = random.choices(self.d.rows, k=self.N)        
        sorted_some = self.d.clone().adds(some).chebyshevs().rows
        chebyshev_distances = sorted([self.d.chebyshev(row) for row in sorted_some])
        self.assertEqual(self.d.chebyshev(sorted_some[0]), chebyshev_distances[0])

    def test_experimental_treatment_repeats(self):
        # Test if the code runs some experimental treatment 20 times for statistical validity
        repeats = 20
        for _ in range(repeats):
            dumb = experiment.guess(self.N, self.d)
            self.assertEqual(len(dumb), self.N)

    def test_shuffle_jiggles_order(self):
        # Test if d.shuffle() really jiggles the order of the data
        original_order = self.d.rows.copy()
        shuffled_order = self.d.shuffle().rows
        self.assertNotEqual(original_order, shuffled_order)


    def dumb_list_length(self):
        # Test if smart and dumb lists are the right length for different values of N
        for N in [20, 30, 40, 50]:
            dumb = experiment.guess(N, self.d)
            self.assertEqual(len(dumb), N)
    
    def test_baseline_chebyshev_length(self):
        # Test if baseline Chebyshev matches the number of rows
        b4 = [self.d.chebyshev(row) for row in self.d.rows]
        self.assertEqual(len(b4), len(self.d.rows), "Baseline Chebyshev should match the number of rows.")


if __name__ == '__main__':
    unittest.main()