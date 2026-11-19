import unittest

from pricing import total


class PricingTests(unittest.TestCase):
    def test_subtracts_discount(self):
        self.assertEqual(total(100, 20), 80)
