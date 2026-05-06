import unittest
from diversity.patterns import Token


class TokenPatternsTest(unittest.TestCase):

  def test_returns_empty_list_for_empty_input(self):
    """Test that empty input returns an empty list."""
    self.assertEqual(Token.token_patterns([], n=2), [])

  def test_returns_empty_for_sentences_shorter_than_n(self):
    """Test that sentences shorter than n produce no patterns."""
    data = ["A B", "C"]
    n = 3
    self.assertEqual(Token.token_patterns(data, n), [])

  def test_extracts_unigrams_correctly(self):
    """Test behavior with n=1 (unigrams)."""
    data = ["A B C"]
    n = 1
    patterns = Token.token_patterns(data, n, top_n=10)
    patterns_dict = dict(patterns)

    self.assertEqual(patterns_dict.get("A"), 1)
    self.assertEqual(patterns_dict.get("B"), 1)
    self.assertEqual(patterns_dict.get("C"), 1)

  def test_counts_pattern_frequencies_correctly(self):
    """Test that it correctly counts multiple occurrences of the same pattern."""
    data = ["A B C", "A B D"]
    n = 2
    patterns = Token.token_patterns(data, n, top_n=10)
    patterns_dict = dict(patterns)

    # "A B" appears twice
    self.assertEqual(patterns_dict.get("A B"), 2)
    # "B C" appears once
    self.assertEqual(patterns_dict.get("B C"), 1)

  def test_does_not_cross_sentence_boundaries(self):
    """Test that patterns are not formed across sentence boundaries.

    This verifies the fix for the bug where patterns like 'C D' were incorrectly
    generated from input ["A B C", "D E F"].
    """
    # Two sentences that, if joined directly, would create "C D"
    data = ["A B C", "D E F"]
    n = 2

    # "C D" should NOT appear if boundaries are respected
    patterns = Token.token_patterns(data, n, top_n=10)
    pattern_strings = [p[0] for p in patterns]

    self.assertNotIn("C D", pattern_strings)
    self.assertIn("A B", pattern_strings)
    self.assertIn("B C", pattern_strings)
    self.assertIn("D E", pattern_strings)
    self.assertIn("E F", pattern_strings)

  def test_sorts_patterns_by_frequency_descending(self):
    """Test that patterns are returned in descending order of frequency."""
    data = ["A B", "C D", "C D", "E F", "E F", "E F"]
    n = 2
    patterns = Token.token_patterns(data, n, top_n=10)

    self.assertEqual(patterns[0], ("E F", 3))
    self.assertEqual(patterns[1], ("C D", 2))
    self.assertEqual(patterns[2], ("A B", 1))

  def test_truncates_results_to_top_n(self):
    """Test that the result is limited to top_n items."""
    # Create 3 distinct patterns: "A B" (3x), "C D" (2x), "E F" (1x)
    data = ["A B", "A B", "A B", "C D", "C D", "E F"]
    n = 2

    # Limit to top 2
    patterns = Token.token_patterns(data, n, top_n=2)

    self.assertEqual(len(patterns), 2)
    self.assertEqual(patterns[0], ("A B", 3))
    self.assertEqual(patterns[1], ("C D", 2))

  def test_respects_case_and_whitespace(self):
    """Test that tokenization is specific (space split) and case sensitive."""
    data = ["The cat", "the Cat"]
    n = 1
    patterns = Token.token_patterns(data, n, top_n=10)
    patterns_dict = dict(patterns)

    # "The" and "the" are distinct
    self.assertEqual(patterns_dict.get("The"), 1)
    self.assertEqual(patterns_dict.get("the"), 1)
    # "cat" and "Cat" are distinct
    self.assertEqual(patterns_dict.get("cat"), 1)
    self.assertEqual(patterns_dict.get("Cat"), 1)


if __name__ == "__main__":
  unittest.main()
