#!/bin/bash
set -e

# Generate the device source and compile it into ./kernel_lib.so via the
# framework's LibraryGenerator (see example_gemm.py), then run the test.
python example_gemm.py
python test_example_gemm.py
