# -*- coding: utf-8 -*-
class FileExtentionError(Exception):
    """Raised when file does not contain the right extension."""

    pass


class GranularityError(Exception):
    """Raised when granularity is not set correctly"""

    pass
