import unittest
import numpy as np

class TestResNet(unittest.TestCase):

    def setUp(self):
		state = np.mat([[1.0,-.5], [0.5,-1.0]])
        pass #self.seq = range(10)

#    def test_fromout_dtanh(self):
        # make sure the shuffled sequence does not lose any elements
#       self.assertEqual(self.seq, range(10))

    def test_choice(self):
        element = random.choice(self.seq)
        self.assertTrue(element in self.seq)

    def test_sample(self):
        self.assertRaises(ValueError, random.sample, self.seq, 20)
        for element in random.sample(self.seq, 5):
            self.assertTrue(element in self.seq)

if __name__ == '__main__':
    unittest.main()

