# -*- coding: utf-8 -*-

"""exceptions.py

Defines exceptions and errors used in the package
"""


class FileExtentionError(Exception):
    """Raised when file does not contain the right extension."""

    ...


class GranularityError(Exception):
    """Raised when granularity is not set correctly"""

    ...


class AdjacencyMatrixError(Exception):
    """Raised when adjacency matrix cannot be parsed into a graph"""

    ...


class GraphTypeError(Exception):
    """Raised when wrong graph type is declared"""

    ...


class TDAPipelineError(Exception):
    """Raised when TDA pipeline is set incorrectly"""

    ...


class ProteinLoadingError(Exception):
    """Raised when protein cannot be correctly loaded from disk"""

    ...


class UniquenessError(Exception):
    """Raised when not enough random strings can be generated from given
    string_length"""

    ...
