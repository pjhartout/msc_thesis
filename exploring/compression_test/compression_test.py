#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""compression_test.py

This script aims to evaluate the ability of different compression algorithms to compress a list of protein objects.

Inspired from: https://stackoverflow.com/questions/57983431/whats-the-most-space-efficient-way-to-compress-serialized-python-data

"""

import bz2
import gzip
import lzma
import pickle

import brotli

from proteinggnnmetrics.utils.debug import measure_memory, timeit
from proteinggnnmetrics.utils.functions import load_obj


@timeit
@measure_memory
def no_compression(data):
    with open(
        "/local0/scratch/phartout/data/data/systematic/representations/human/unperturbed/no_compression.pickle",
        "wb",
    ) as f:
        pickle.dump(data, f)


@timeit
@measure_memory
def gzip_c(data):
    with gzip.open(
        "/local0/scratch/phartout/data/data/systematic/representations/human/unperturbed/gzip_test.gz",
        "wb",
    ) as f:
        pickle.dump(data, f)


@timeit
@measure_memory
def bz2_c(data):
    with bz2.BZ2File(
        "/local0/scratch/phartout/data/data/systematic/representations/human/unperturbed/bz2_test.pbz2",
        "wb",
    ) as f:
        pickle.dump(data, f)


@timeit
@measure_memory
def lzma_c(data):
    with lzma.open(
        "/local0/scratch/phartout/data/data/systematic/representations/human/unperturbed/lmza_test.xz",
        "wb",
    ) as f:
        pickle.dump(data, f)


@timeit
@measure_memory
def brotli_c(data):
    with open(
        "/local0/scratch/phartout/data/data/systematic/representations/human/unperturbed/no_compression.pickle",
        "rb",
    ) as f:
        pdata = f.read()
        with open(
            "/local0/scratch/phartout/data/data/systematic/representations/human/unperturbed/brotli_test.bt",
            "wb",
        ) as b:
            b.write(brotli.compress(pdata))


def main():
    data = load_obj(
        "/local0/scratch/phartout/data/data/systematic/representations/human/unperturbed/constant_reps.pkl"
    )
    no_compression(data)
    gzip_c(data)
    bz2_c(data)
    lzma_c(data)
    brotli_c(data)


if __name__ == "__main__":
    main()
