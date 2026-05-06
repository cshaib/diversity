from typing import List, Tuple
import unittest
from diversity.patterns import part_of_speech


# Mock data for unit testing pos_patterns without SpaCy
def create_pos_tuples(
    sentences: List[str], tags: List[str]
) -> List[List[Tuple[str, str]]]:
  result = []
  words = [s.split() for s in sentences]
  tag_lists = [t.split() for t in tags]
  for w_list, t_list in zip(words, tag_lists):
    result.append(list(zip(w_list, t_list)))
  return result


class PartOfSpeechTest(unittest.TestCase):

  def test_pos_patterns_basic(self):
    text = [["The", "quick", "fox"]]
    tags = [["DT", "JJ", "NN"]]
    data = [[(w, t) for w, t in zip(text[0], tags[0])]]

    matches = part_of_speech.pos_patterns(data, "DT JJ NN")
    self.assertEqual(matches, {"The quick fox"})

  def test_pos_patterns_no_match(self):
    text = [["The", "quick", "fox"]]
    tags = [["DT", "JJ", "NN"]]
    data = [[(w, t) for w, t in zip(text[0], tags[0])]]

    matches = part_of_speech.pos_patterns(data, "NN NN")
    self.assertEqual(matches, set())

  def test_pos_patterns_partial_match(self):
    sentences = ["The quick brown fox"]
    tags = ["DT JJ NN NN"]
    data = create_pos_tuples(sentences, tags)

    matches = part_of_speech.pos_patterns(data, "JJ NN")
    self.assertIn("quick brown", matches)
    matches = part_of_speech.pos_patterns(data, "NN NN")
    self.assertIn("brown fox", matches)

  def test_pos_patterns_multiple_occurrences(self):
    sentences = ["A cat and a dog"]
    tags = ["DT NN CC DT NN"]
    data = create_pos_tuples(sentences, tags)

    matches = part_of_speech.pos_patterns(data, "DT NN")
    self.assertEqual(matches, {"A cat", "a dog"})

  def test_pos_patterns_with_newline(self):
    # Manually construct data because create_pos_tuples uses split() which
    # removes newlines
    data = [[("A", "DT"), ("\n", "SPACE"), ("B", "NN")]]

    matches = part_of_speech.pos_patterns(data, "DT SPACE NN")
    self.assertEqual(matches, {"A \n B"})

  def test_pos_patterns_cross_document_boundary(self):
    # Doc 1: "The cat" (DT NN)
    # Doc 2: "sat on" (VBD IN)
    data = [[("The", "DT"), ("cat", "NN")], [("sat", "VBD"), ("on", "IN")]]
    matches = part_of_speech.pos_patterns(data, "NN VBD")
    self.assertEqual(matches, set())

  def test_pos_patterns_long_sequence(self):
    data = [[
        ("A", "DT"),
        ("\n", "SPACE"),
        ("long", "JJ"),
        ("\t", "SPACE"),
        ("sentence", "NN"),
        ("with", "IN"),
        ("many", "JJ"),
        ("words", "NNS"),
        (".", "."),
    ]]
    matches = part_of_speech.pos_patterns(data, "JJ SPACE NN")
    self.assertEqual(matches, {"long \t sentence"})


if __name__ == "__main__":
  unittest.main()
