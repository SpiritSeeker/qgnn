"""
Module to parse QASM code and convert it into a graph representation

Functions
---------
get_encoding(gate: str, theta = None)
    Get the encoding for a given gate

check_valid_gate(gate: str)
    Check if the gate is a valid gate

preprocess(qasm: List[str])
    Preprocess the QASM code

preprocess_gates(qasm: List[str])
    Preprocess the gate definitions in the QASM code

get_gate_info(gate_def)
    Get the gate name and body from the gate definition

replace_gate(qasm_str, gate_name, gate_body)
    Replace the gate in the QASM code with the gate body

parse(qasm: List[str])
    Parse the QASM code and return the nodes and edges of the graph
"""

from typing import List

import numpy as np

GATE_IDX = {
    'input': 0,
    'measure': 1,
    'reset': 2,
    'cx': 3,
    'id': 4,
    'x': 5,
    'y': 6,
    'z': 7,
    'h': 8,
    's': 9,
    'sdg': 10,
    't': 11,
    'tdg': 12,
    'barrier': 13,
}

NODE_EMBEDDING_DIM = 17
SINGLE_GATES = ['reset', 'id', 'x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg']
ROTATION_GATES = ['rx', 'ry', 'rz', 'u1', 'u2', 'u3', 'u']
CONTROL_GATES = ['cx']
VALID_GATES = SINGLE_GATES + CONTROL_GATES + ROTATION_GATES + ['barrier', 'measure']

def get_encoding(gate: str, theta = None):
    """
    Get the encoding for a given gate

    Parameters
    ----------
    gate : str
        Name of the gate

    theta : np.ndarray, optional
        Rotation angles for the gate
        Only required for rotation gates

    Returns
    -------
    np.ndarray
        One-hot encoding of the gate

    Raises
    ------
    ValueError
        If the gate is invalid

    Notes
    -----
    * For all the gates present in the GATE_IDX dictionary, the encoding is a one-hot encoding
    * Rotation gates with standard angles are encoded as the corresponding standard gate
    * Rotation gates with non-standard angles are encoded as a 3-dimensional vector with the angles
        - sin(theta[0]), sin(theta[1]), sin(theta[2])
    * The first 14 dimensions are for the standard gates
    * The last 3 dimensions are for the rotation angles
    """
    encoding = np.zeros(NODE_EMBEDDING_DIM)
    if gate in GATE_IDX:
        encoding[GATE_IDX[gate]] = 1
    elif gate in ROTATION_GATES:
        rotation_angles = np.zeros(3)
        if theta is None:
            raise ValueError(f"Theta not provided for gate {gate}")
        if gate == 'rx':
            rotation_angles[0] = theta[0]
            rotation_angles[1] = -np.pi/2
            rotation_angles[2] = np.pi/2
        elif gate == 'ry':
            rotation_angles[0] = theta[0]
        elif gate == 'rz':
            if np.allclose(theta[0], np.pi/2):
                encoding[GATE_IDX['s']] = 1
            elif np.allclose(theta[0], -np.pi/2):
                encoding[GATE_IDX['sdg']] = 1
            elif np.allclose(theta[0], np.pi/4):
                encoding[GATE_IDX['t']] = 1
            elif np.allclose(theta[0], -np.pi/4):
                encoding[GATE_IDX['tdg']] = 1
            elif np.allclose(theta[0], np.pi):
                encoding[GATE_IDX['z']] = 1
            elif np.allclose(theta[0], 0):
                encoding[GATE_IDX['id']] = 1
            else:
                rotation_angles[2] = theta[0]
        elif gate in ['u1', 'u2', 'u3', 'u']:
            is_standard, val = check_standard_rotation(gate, theta)
            if is_standard:
                encoding[GATE_IDX[val]] = 1
            else:
                rotation_angles = np.array(val)
        else:
            raise ValueError(f"Invalid gate: {gate}")
        encoding[-3:] = np.sin(rotation_angles)
    return encoding

def check_standard_rotation(gate: str, theta):
    """
    Check if the rotation gate has standard angles

    Parameters
    ----------
    gate : str
        Name of the gate

    theta : np.ndarray
        Rotation angles

    Returns
    -------
    Tuple[bool, Union[str, np.ndarray]]
        Tuple containing a boolean value and the standard gate name or the rotation angles
        True if the angles are standard, False otherwise
    """
    if gate == 'u1':
        if np.allclose(theta[0], 0):
            return True, 'id'
        if np.allclose(theta[0], np.pi):
            return True, 'z'
        if np.allclose(theta[0], np.pi/2):
            return True, 's'
        if np.allclose(theta[0], -np.pi/2):
            return True, 'sdg'
        if np.allclose(theta[0], np.pi/4):
            return True, 't'
        if np.allclose(theta[0], -np.pi/4):
            return True, 'tdg'
        return False, [0, 0, theta[0]]
    if gate == 'u2':
        if np.allclose(theta, np.array([0, np.pi])):
            return True, 'h'
        return False, [np.pi/2, theta[0], theta[1]]
    if gate == 'u3' or gate == 'u':
        if np.allclose(theta, np.array([0, 0, 0])):
            return True, 'id'
        if np.allclose(theta, np.array([np.pi, 0, np.pi])):
            return True, 'x'
        if np.allclose(theta, np.array([np.pi, np.pi/2, np.pi/2])):
            return True, 'y'
        if np.allclose(theta, np.array([0, 0, np.pi/2])):
            return True, 'z'
        if np.allclose(theta, np.array([np.pi/2, 0, np.pi])):
            return True, 'h'
        if np.allclose(theta, np.array([0, 0, np.pi/2])):
            return True, 's'
        if np.allclose(theta, np.array([0, 0, -np.pi/2])):
            return True, 'sdg'
        if np.allclose(theta, np.array([0, 0, np.pi/4])):
            return True, 't'
        if np.allclose(theta, np.array([0, 0, -np.pi/4])):
            return True, 'tdg'
        return False, theta

def check_valid_gate(gate: str):
    """
    Check if the gate is a valid gate
    """
    if '(' in gate:
        gate = gate.split('(')[0]
    return gate in VALID_GATES

def preprocess(qasm):
    """
    Preprocess the QASM code

    Parameters
    ----------
    qasm : List[str]
        List of lines of the QASM code

    Returns
    -------
    List[str]
        List of preprocessed lines of the QASM code

    Notes
    -----
    * The QASM code is preprocessed to remove comments, headers, and replace gate definitions
    * Standard gate definitions are used from 'qgnn/qasm/qelib.inc'
    """
    with open('qgnn/qasm/qelib.inc', 'r') as f:
        header = f.readlines()
    qasm = header + qasm
    qasm = [line.strip() for line in qasm]

    # Remove headers
    qasm = [line for line in qasm if line and not line.startswith('//')]
    qasm = [line for line in qasm if line and not line.startswith('OPENQASM')]
    qasm = [line for line in qasm if line and not line.startswith('include')]

    # Remove comments from every line
    qasm = [line.split('//')[0] for line in qasm]

    # Remove any number of leading spaces before (, if any
    qasm = [line.replace(' (', '(') for line in qasm]

    # Break multiple statements in a line into separate lines
    qasm = '\n'.join(qasm)
    qasm = qasm.split(';')
    qasm = [line.strip() for line in qasm]

    # Check if there is a gate definition
    has_gate = False
    for line in qasm:
        if line.startswith('gate'):
            has_gate = True
            break

    if has_gate:
        qasm = preprocess_gates(qasm)


    # Remove empty lines
    qasm = [line for line in qasm if line]

    # Trim all lines till semicolon
    qasm = [line.split(';')[0] for line in qasm]

    # Remove trailing whitespaces
    qasm = [line.strip() for line in qasm]
    return qasm

def preprocess_gates(qasm):
    """
    Preprocess the gate definitions in the QASM code

    Parameters
    ----------
    qasm : List[str]
        List of lines of the QASM code

    Returns
    -------
    List[str]
        List of preprocessed lines of the QASM code

    Notes
    -----
    * Gate definitions are read and replaced in the QASM code
    * Gate definitions are removed from the QASM code
    """
    qasm_str = '\n'.join(qasm)
    qasm_str = '\n' + qasm_str

    while True:
        # Find location of gate definition which starts at newline
        gate_start = qasm_str.find('\ngate')
        if gate_start == -1:
            break
        gate_end = qasm_str.find('}', gate_start)
        if gate_end == -1:
            raise ValueError("Invalid gate definition")

        gate_name, gate_body = get_gate_info(qasm_str[gate_start:gate_end+1])

        # Remove gate definition from qasm string
        qasm_str = qasm_str[:gate_start] + qasm_str[gate_end+1:]

        qasm_str = replace_gate(qasm_str, gate_name, gate_body)

    return qasm_str.split('\n')

def get_gate_info(gate_def):
    """
    Get the gate name and body from the gate definition

    Parameters
    ----------
    gate_def : str
        Gate definition

    Returns
    -------
    Tuple[str, str]
        Tuple containing the gate name and the gate body

    Notes
    -----
    * The gate body is modified to replace the gate parameters and targets with placeholders
    """
    gate_header = gate_def.split('{')[0].strip()

    gate_name = gate_header.split()[1]
    # Check if gate has parameters
    gate_params = []
    if gate_header.find('(') != -1:
        gate_params = gate_header.split('(')[1].split(')')[0]
        gate_params = gate_params.split(',')
        gate_params = [gp.strip() for gp in gate_params]
        gate_name = gate_name.split('(')[0]

    # Parse gate targets
    if ')' in gate_header:
        gate_targets = gate_header.split(')')[1].strip()
    else:
        gate_targets = gate_header.split()[2:]
    gate_targets = ' '.join(gate_targets).split(',')
    gate_targets = [gt.strip() for gt in gate_targets]
    gate_targets = [gt.replace(' ', '') for gt in gate_targets]

    gate_body = gate_def.split('{')[1].split('}')[0]
    gate_body = gate_body.strip()
    gate_body = gate_body.split('\n')
    gate_body = [line.strip() for line in gate_body]
    for li in range(len(gate_body)):
        line = gate_body[li]
        # Check if the gate in the line has a parameter
        if line.find('(') != -1:
            # Grab content of the outermost parentheses
            first_paran = line.find('(')
            last_paran = line.rfind(')')
            g_param = line[first_paran+1:last_paran]
            g_param = g_param.strip()
            g_param = g_param.split(',')
            g_param = [gp.strip() for gp in g_param]
            g_name = line.split('(')[0]
            for i in range(len(g_param)):
                for p in gate_params:
                    if p in g_param[i]:
                        # Check if p is surrounded by letters
                        if g_param[i].find(p) == 0:
                            if len(p) < len(g_param[i]):
                                if g_param[i][len(p)] not in [')', ',', ' ', '*', '+', '-', '/', '^', '%']:
                                    continue
                        else:
                            if g_param[i][g_param[i].find(p)-1] not in ['(', ',', ' ', '*', '+', '-', '/', '^', '%']:
                                continue
                        g_param[i] = g_param[i].replace(p, f"__p{gate_params.index(p)}")
            g_param = ','.join(g_param)
            line = f"{g_name}({g_param}){line[last_paran+1:]}"

        # Gate targets
        g_targets = line.split()[1:]
        g_targets = ' '.join(g_targets).split(',')
        g_targets = [gt.strip() for gt in g_targets]

        # Remove trailing semicolon, if present
        if g_targets[-1].endswith(';'):
            g_targets[-1] = g_targets[-1][:-1]

        for i in range(len(g_targets)):
            if g_targets[i] in gate_targets:
                g_targets[i] = f"__t{gate_targets.index(g_targets[i])}"
        g_targets = ','.join(g_targets)
        line = f"{line.split()[0]} {g_targets};"
        gate_body[li] = line

    gate_body = '\n'.join(gate_body)

    return gate_name, gate_body

def replace_gate(qasm_str, gate_name, gate_body):
    """
    Replace the gate in the QASM code with the gate body

    Parameters
    ----------
    qasm_str : str
        QASM code

    gate_name : str
        Name of the gate to be replaced

    gate_body : str
        Body of the gate

    Returns
    -------
    str
        QASM code with the gate replaced

    Notes
    -----
    * The gate name and targets are replaced in the gate body
    """
    qasm_str = qasm_str.split('\n')
    for i in range(len(qasm_str)):
        line = qasm_str[i]
        if not line:
            continue
        # Find the gate name in the line
        g_name = line.split()[0]
        if '(' in g_name:
            g_name = g_name.split('(')[0]
        if g_name == gate_name:
            # Copy gate_body to g_body
            g_body = gate_body

            # Extract gate params, if any
            if line.find('(') != -1:
                gate_params = line.split('(')[1].split(')')[0]
                gate_params = gate_params.split(',')
                gate_params = [gp.strip() for gp in gate_params]
                for j in range(len(gate_params)):
                    g_body = g_body.replace(f"__p{j}", gate_params[j])

            # Extract gate targets
            gate_targets = line.split()[1:]
            gate_targets = ' '.join(gate_targets).split(',')
            gate_targets = [gt.strip() for gt in gate_targets]
            if gate_targets[-1].endswith(';'):
                gate_targets[-1] = gate_targets[-1][:-1]
            for j in range(len(gate_targets)):
                g_body = g_body.replace(f"__t{j}", gate_targets[j])

            qasm_str[i] = g_body
    return '\n'.join(qasm_str)

def parse(qasm: List[str]):
    """
    Parse the QASM code and return the nodes and edges of the graph

    Parameters
    ----------
    qasm : List[str]
        List of lines of the QASM code

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing the nodes and edges of the graph

    Raises
    ------
    ValueError
        If an invalid gate is encountered
        If a dynamic circuit is encountered

    Notes
    -----
    * The QASM code is preprocessed before parsing
    * The input qubits are parsed first
    * Each input, measurement, and gate is encoded as a node
    * The edges are encoded as connections between the nodes
    * The control qubits of a CNOT gate are an edge to the CNOT gate node and also the next gate in the circuit
        - The CNOT gate node is a single output node
    * Dynamic circuits are currently not supported
        - The code will raise a ValueError if a dynamic circuit is encountered
    """
    qasm = preprocess(qasm)

    nodes = []
    edges = []
    input_qubits = {}
    current_qubit_node = {}
    measure_nodes = {}
    for line in qasm:
        # Parse input qubits
        if line.startswith('qreg'):
            # Infer name of qubit and number of qubits
            qubit_name = line.split()[1].split('[')[0]
            n_qubits = int(line.split()[1].split('[')[1].split(']')[0])
            input_qubits[qubit_name] = n_qubits
            for i in range(n_qubits):
                nodes.append(get_encoding('input'))
                current_qubit_node[f"{qubit_name}[{str(i)}]"] = len(nodes) - 1
            continue

        if line.startswith('creg'):
            name = line.split()[1].split('[')[0]
            n_qubits = int(line.split()[1].split('[')[1].split(']')[0])
            for i in range(n_qubits):
                nodes.append(get_encoding('measure'))
                measure_nodes[f"{name}[{str(i)}]"] = len(nodes) - 1
            continue

        gate = line.split()[0]
        if gate.startswith('if('):
            raise ValueError("Dynamic circuits not supported")
        if not check_valid_gate(gate):
            raise ValueError(f"Invalid gate: {gate}")

        if gate == 'measure':
            # Remove leading measure and leading/trailing whitespaces
            line_chunk = line[7:].strip()
            line_chunk = line_chunk.split('->')
            line_chunk = [lc.strip() for lc in line_chunk]
            # Remove trailing semicolon, if present
            if line_chunk[1].endswith(';'):
                line_chunk[1] = line_chunk[1][:-1]
            if line_chunk[1] in measure_nodes and line_chunk[0] in current_qubit_node:
                edges.append((current_qubit_node[line_chunk[0]], measure_nodes[line_chunk[1]]))
                current_qubit_node[line_chunk[0]] = measure_nodes[line_chunk[1]]
            elif line_chunk[0] in input_qubits:
                for i in range(input_qubits[line_chunk[0]]):
                    edges.append((current_qubit_node[f"{line_chunk[0]}[{str(i)}]"], measure_nodes[f"{line_chunk[1]}[{str(i)}]"]))
                    current_qubit_node[f"{line_chunk[0]}[{str(i)}]"] = measure_nodes[f"{line_chunk[1]}[{str(i)}]"]
            else:
                raise ValueError(f"Qubit {line_chunk[0]} not defined")
            continue

        if gate in SINGLE_GATES:
            qubit = line.split()[1]
            if qubit.endswith(';'):
                qubit = qubit[:-1]
            if qubit in current_qubit_node:
                nodes.append(get_encoding(gate))
                edges.append((current_qubit_node[qubit], len(nodes) - 1))
                current_qubit_node[qubit] = len(nodes) - 1
            elif qubit in input_qubits:
                for i in range(input_qubits[qubit]):
                    nodes.append(get_encoding(gate))
                    edges.append((current_qubit_node[f"{qubit}[{str(i)}]"], len(nodes) - 1))
                    current_qubit_node[f"{qubit}[{str(i)}]"] = len(nodes) - 1
            else:
                raise ValueError(f"Qubit {qubit} not defined")
            continue

        if gate in CONTROL_GATES:
            line_chunk = line[3:].strip()
            line_chunk = line_chunk.split(',')
            line_chunk = [lc.strip() for lc in line_chunk]
            if line_chunk[1].endswith(';'):
                line_chunk[1] = line_chunk[1][:-1]
            nodes.append(get_encoding(gate))
            edges.append((current_qubit_node[line_chunk[0]], len(nodes) - 1))
            edges.append((current_qubit_node[line_chunk[1]], len(nodes) - 1))
            # current_qubit_node[line_chunk[0]] = len(nodes) - 1
            current_qubit_node[line_chunk[1]] = len(nodes) - 1
            continue

        if gate == 'barrier':
            # Remove leading barrier and leading/trailing whitespaces
            line_chunk = line[7:].strip()
            line_chunk = line_chunk.split(',')
            line_chunk = [lc.strip() for lc in line_chunk]
            for qubit in line_chunk:
                if qubit.endswith(';'):
                    qubit = qubit[:-1]
                if qubit in current_qubit_node:
                    nodes.append(get_encoding(gate))
                    edges.append((current_qubit_node[qubit], len(nodes) - 1))
                    current_qubit_node[qubit] = len(nodes) - 1
                elif qubit in input_qubits:
                    for i in range(input_qubits[qubit]):
                        nodes.append(get_encoding(gate))
                        edges.append((current_qubit_node[f"{qubit}[{str(i)}]"], len(nodes) - 1))
                        current_qubit_node[f"{qubit}[{str(i)}]"] = len(nodes) - 1
                else:
                    raise ValueError(f"Qubit {qubit} not defined")
            continue

        if '(' in gate:
            # Get gate name and parameters
            gate_name = gate.split('(')[0]
            gate_params = gate[gate.find('(')+1:gate.rfind(')')]
            gate_params = gate_params.split(',')
            gate_params = [gp.strip() for gp in gate_params]
            gate_params = '[' + ', '.join(gate_params) + ']'
            pi = np.pi
            gate_params = eval(gate_params)
            gate_params = np.array(gate_params)

            # Get gate targets
            gate_targets = line.split()[1:]
            gate_targets = ' '.join(gate_targets).split(',')
            gate_targets = [gt.strip() for gt in gate_targets]
            if gate_targets[-1].endswith(';'):
                gate_targets[-1] = gate_targets[-1][:-1]

            for target in gate_targets:
                if target in current_qubit_node:
                    nodes.append(get_encoding(gate_name, gate_params))
                    edges.append((current_qubit_node[target], len(nodes) - 1))
                    current_qubit_node[target] = len(nodes) - 1
                elif target in input_qubits:
                    for i in range(input_qubits[target]):
                        nodes.append(get_encoding(gate_name, gate_params))
                        edges.append((current_qubit_node[f"{target}[{str(i)}]"], len(nodes) - 1))
                        current_qubit_node[f"{target}[{str(i)}]"] = len(nodes) - 1
                else:
                    raise ValueError(f"Qubit {target} not defined")


    return np.array(nodes), np.array(edges)
