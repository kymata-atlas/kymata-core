from kymata.math.combinatorics import generate_derangement

def test_derangement_length(self):
    n = 10
    derangement = generate_derangement(n)
    self.assertEqual(len(derangement), n)


def test_is_derangement(self):
    n = 10
    derangement = generate_derangement(n)
    for i in range(n):
        self.assertNotEqual(derangement[i], i, f"Element at index {i} is not a derangement")


def test_single_element(self):
    n = 1
    derangement = generate_derangement(n)
    self.assertEqual(len(derangement), n)
    self.assertEqual(derangement[0], 0, "Single element should remain as is")


def test_larger_derangement(self):
    n = 1000
    derangement = generate_derangement(n)
    self.assertEqual(len(derangement), n)
    for i in range(n):
        self.assertNotEqual(derangement[i], i, f"Element at index {i} is not a derangement")


def test_repeatability(self):
    n = 10
    derangement1 = generate_derangement(n)
    derangement2 = generate_derangement(n)
    self.assertEqual(len(derangement1), len(derangement2))
    self.assertNotEqual(derangement1.tolist(), derangement2.tolist(), "Derangements should not be identical")
