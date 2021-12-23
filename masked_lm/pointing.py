class Point(object):
  """Point that corresponds to a token edit operation.

  Attributes:
    point_index: The index of the next token in the sequence.
    added_phrase: A phrase that's inserted before the next token (can be empty).
  """

  def __init__(self, point_index, added_phrase=''):
    """Constructs a Point object .

    Args:
      point_index: The index the of the next token in the sequence.
      added_phrase: A phrase that's inserted before the next token.

    Raises:
      ValueError: If point_index is not an Integer.
    """

    self.added_phrase = added_phrase

    try:
      self.point_index = int(point_index)
    except ValueError:
      raise ValueError(
          'point_index should be an Integer, not {}'.format(point_index))

  def __str__(self):
    return '{}|{}'.format(self.point_index, self.added_phrase)