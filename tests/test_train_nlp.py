import unittest
from collections import Counter

from train_nlp import split_single_label_indices


class NlpSplitTests(unittest.TestCase):
    def test_singleton_label_remains_in_training(self):
        labels = ([0] * 20) + ([1] * 20) + ([2] * 1)

        train_ids, validation_ids, test_ids, rare = split_single_label_indices(
            labels, seed=42
        )

        self.assertEqual(rare, {2: 1})
        self.assertIn(40, train_ids)
        self.assertNotIn(40, validation_ids)
        self.assertNotIn(40, test_ids)
        self.assertEqual(
            set(train_ids) | set(validation_ids) | set(test_ids),
            set(range(len(labels))),
        )
        self.assertFalse(set(train_ids) & set(validation_ids))
        self.assertFalse(set(train_ids) & set(test_ids))
        self.assertFalse(set(validation_ids) & set(test_ids))
        self.assertEqual(Counter(labels[index] for index in validation_ids), {0: 2, 1: 2})
        self.assertEqual(Counter(labels[index] for index in test_ids), {0: 2, 1: 2})


if __name__ == "__main__":
    unittest.main()
