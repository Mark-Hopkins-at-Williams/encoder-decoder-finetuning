import unittest
from permutations import create_random_permutation_with_fixed_points


class TestPermutations(unittest.TestCase):
    def test_permutations_identity(self):
        pmap = create_random_permutation_with_fixed_points(5, fixed_points=[0, 1, 2, 3, 4])
        for i in range(5):
            self.assertEqual(pmap(i), i)
            
    def test_permutation_random(self):
        vocab_size = 5
        fixed = [1, 3, 4]
        pmap = create_random_permutation_with_fixed_points(vocab_size, fixed)
        for i in fixed:
            self.assertEqual(pmap(i), i)
        permuted = [pmap(i) for i in range(vocab_size) if i not in fixed]
        expected = set(range(vocab_size)) - set(fixed)
        self.assertCountEqual(permuted, expected)
        
    def test_permutation_call(self):
        pmap = create_random_permutation_with_fixed_points(3, [])
        self.assertIsInstance(pmap(0), int)
        
        
if __name__ == "__main__":
    unittest.main()
