"""
Module for reading QASM files.

Functions
---------
read_qasm_file(file_path)
    Read a QASM file and return the parsed data.
"""

import os
from typing import Tuple

import numpy as np

from . import qasm_parser


def read_qasm_file(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a QASM file and return the parsed data.

    Arguments
    ---------
    file_path : str
        Path to the QASM file.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple of numpy arrays containing the nodes and edges of the graph.

    Raises
    ------
    FileNotFoundError
        If the file is not found.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r') as f:
        qasm_str = f.readlines()

    return qasm_parser.parse(qasm_str)
