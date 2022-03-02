# -*- coding: utf-8 -*-

"""exceptions.py

Defines exceptions and errors used in the package
"""


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


class TDAPipelineError(Exception):
    """Raised when TDA pipeline is set incorrectly"""

    pass


class ProteinLoadingError(Exception):
    """Raised when protein cannot be correctly loaded from disk"""

    pass
