# -*- coding: utf-8 -*-
class FileExtentionError(Exception):
    """Raised when file does not contain the right extension."""

    pass


class GranularityError(Exception):
    """Raised when granularity is not set correctly"""

    pass


class AdjacencyMatrixError(Exception):
    """Raised when adjacency matrix cannot be parsed into a graph"""

    pass


class GraphTypeError(Exception):
    """Raised when wrong graph type is declared"""

    pass
