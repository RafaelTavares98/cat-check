"""The one place a picture becomes numbers.

Training and judging call this same function. A second copy of these
three lines somewhere else is how a model comes to be measured on
features production will never build.
"""

import numpy


def build(pictures, top=255.0):
    """Rows of pixels, scaled to sit between 0 and 1.

    `top` is the brightest value the source can produce: 255 for a
    photograph, 16 for the digits. Passing the wrong one does not break
    the model, it quietly changes what every number means, so it is an
    argument and never a guess.
    """
    array = numpy.asarray(pictures, dtype=float)
    _demand_rows(array, top)
    return array / top


def _demand_rows(array, top):
    if array.ndim != 2:
        raise ValueError(
            f"pictures arrive as one row each, and these have {array.ndim} "
            "dimensions"
        )
    if array.size and array.max() > top:
        raise ValueError(
            f"a pixel reads {array.max():.0f}, brighter than the {top:.0f} "
            "this source is supposed to produce"
        )
