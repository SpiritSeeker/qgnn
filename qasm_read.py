"""
Script to read qasm files and save them as numpy files

Functions
---------
main()
    Main function to read qasm files and save them as numpy files

main_debug()
    Debug function to read a single qasm file and print the result

Comment/Uncomment the function call at the end of the file to run the desired function

Parameters
----------
DATAFILE : str
    Path to the csv file containing the list of qasm files

DATAPREFIX : str
    Prefix to the qasm file paths in the csv file

SAVE_GRAPH_DATA : bool
    If True, save the numpy files for the graphs' nodes and edges
    Saves the files in DATAFILE/numpy directory (creates the directory if it doesn't exist)
"""

import csv
import os

import numpy as np
from tqdm import tqdm

from qgnn.qasm import read_qasm

DATAFILE = 'data/data.csv'
DATAPREFIX = 'data/mqt'
SAVE_GRAPH_DATA = True

def main():
    with open(DATAFILE, mode='r') as f:
        reader = csv.reader(f)
        reader = list(reader)[1:]
        file_list = [row[0] for row in reader]

        # Remove leading /, if any
        file_list = [f[1:] if f[0] == '/' else f for f in file_list]


    # Read qasm files
    not_found_errors = 0
    value_errors = 0
    exceptions = 0
    largest_nodes = 0
    largest_file = ''
    dirname = os.path.join(DATAPREFIX, 'numpy')
    os.makedirs(dirname, exist_ok=True)
    error_list = []
    for file in tqdm(file_list):
        filename = os.path.join(DATAPREFIX, file)
        try:
            qasm = read_qasm.read_qasm_file(filename)

        except FileNotFoundError as e:
            error_list.append(f"File not found: {filename}")
            not_found_errors += 1
            continue
        except ValueError as e:
            error_list.append(f"Error reading file: {filename} - {e}")
            value_errors += 1
            continue
        except Exception as e:
            error_list.append(f"Error reading file: {filename} - {e}")
            exceptions += 1
            continue

        nodes = qasm[0].shape[0]
        if nodes > largest_nodes:
            largest_nodes = nodes
            largest_file = filename

        if SAVE_GRAPH_DATA:
            # Save numpy files
            np.save(os.path.join(dirname, f'{os.path.splitext(file)[0].replace("/","_")}_nodes.npy'), qasm[0])
            np.save(os.path.join(dirname, f'{os.path.splitext(file)[0].replace("/","_")}_edges.npy'), qasm[1])

    if len(error_list) > 0:
        print("Errors:")
        for error in error_list:
            print(error)
    print(f"Number of files not found: {not_found_errors}")
    print(f"Number of value errors: {value_errors}")
    print(f"Number of exceptions: {exceptions}")
    print(f"Total number of files: {len(file_list)}")
    print(f"Largest number of nodes: {largest_nodes}")
    print(f"Largest file: {largest_file}")

def main_debug():
    qasm = read_qasm.read_qasm_file('data/qasm/large/dnn_n33/dnn_n33.qasm')
    print(qasm)


if __name__ == '__main__':
    main()
    # main_debug()
