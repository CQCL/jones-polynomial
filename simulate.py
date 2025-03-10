import argparse
import rich_argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=rich_argparse.RichHelpFormatter)
    parser.add_argument("SPEC", help="machine specification file")
    parser.add_argument("SHOTS", help="number of shots per classical bit string input", type=int)
    parser.add_argument("CIRCUIT", help="quantum circuit file")
    parser.add_argument("-g", "--gpu", help="index of gpus to use for simulation", type=int, nargs="*")
    parser.add_argument("-b", "--bits", help="2D npy array containing classical bit string inputs")
    parser.add_argument("-o", "--output", help="name of measurement output file")
    parser.add_argument("-s", "--scale", help="scale noise rate by a constant factor", type=float, default=1.0)
    parser.add_argument("-e", "--expval", help="measure an expectation value per trajectory")
    parser.add_argument("-B", "--batchsize", help="maximum number of classical bitstrings per batch", type=int, default=256)
    args = parser.parse_args()

import numpy as np
import qiskit
import qiskit.circuit
import qiskit.quantum_info
import qiskit_aer
import json
import cmath
import string
import itertools
import pathlib
import pytket
import pytket.qasm
import tqdm

# Get the unitary matrix for a native gate
def get_unitary(spec: dict, gate: str, params: "list[float]"):
    unitary = spec["gates"][gate]["unitary"]
    q = spec["gates"][gate]["qubits"]
    if isinstance(unitary, dict):
        # Otherwise it is a dictionary with a code attribute
        params = { l: p for l, p in zip(string.ascii_lowercase, params) }
        # Evaluate the code with the parameters as locals to get the matrix
        val = eval(unitary["code"], cmath.__dict__.copy(), params)
        return np.array(val)
    elif isinstance(unitary, list):
        # It could also be just a matrix
        return np.array(unitary)
    else:
        raise ValueError()

# Get the Pauli channel corresponding to the noise for
# a given native gate, or None if there is no noise.
def get_noise(spec: dict, gate: str, params: "list[float]", scale: float): 
    if "error" not in spec["gates"][gate]:
        return
    
    rate = spec["gates"][gate]["error"]["rate"]
    channel = spec["gates"][gate]["error"]["channel"]

    if isinstance(rate, float):
        pass
    elif isinstance(rate, dict):
        # Rate may depend on the parameters, in which case
        # we will have a dictionary with code to eval
        params = { l: p for l, p in zip(string.ascii_lowercase, params) }
        rate = eval(rate["code"], cmath.__dict__.copy(), params)
    else:
        raise ValueError()

    rate *= scale

    q = spec["gates"][gate]["qubits"]
    if isinstance(channel, str):
        # The channel can be a preset given a string
        if channel == "depolarizing":
            # Depolarizing is an equal weighting of all Pauli strings except I..I
            channel = {
                ''.join(p): 1/(4**q - 1) for p in itertools.product('IXYZ', repeat=q) if p != ('I',)*q
            }
        elif channel == "dephasing":
            # Dephasing is only Z errors. How does this generalize?
            assert q == 1
            channel = { 'Z': 1.0 }
        else:
            raise ValueError()
    elif isinstance(channel, dict):
        # You can give custom channels as dictionary directly
        channel = channel.copy()
    else:
        raise ValueError()

    # Flatten the channel to a list of operators making
    # sure that the identity is the first in the list
    ops = ['I'*q]
    probs = [1.0 - rate]
    for p, r in channel.items():
        ops.append(p)
        # Calculate error rate from relative rate and total rate
        probs.append(r * rate)

    return ops, probs

# Convert a pytket circuit in a reasonable format to a list of operations
# We must have the following:
#  1. Exactly one qubit register, at most one classical register
#  2. Every conditional operation is conditioned on exactly one bit
#  3. Every underlying gate must be a native gate
#  4. Every gate has to match the number of qubits and parameters in the spec
#  5. Symbolic parameters and multi-qubit arguments are disallowed
#  6. The maximum used qubit index must be less than the value in the spec
def tket_to_ops(spec: dict, circ: pytket.circuit.Circuit):
    assert len(circ.q_registers) == 1
    assert len(circ.c_registers) <= 1

    ops = []
    nqubits = 0
    for command in circ.get_commands():
        op, args = command.op, command.args
        cbit, cond = None, None

        # If this is a conditional, 
        # extract the classical bit and comparison value
        if isinstance(op, pytket.circuit.Conditional):
            assert op.width == 1
            assert len(args[0].index) == 1
            cbit = args[0].index[0]
            cond = op.value
            op = op.op
            # We assume the classical condition is always
            # first. Is this true in general?
            args = args[1:]

        assert isinstance(op, pytket.circuit.CustomGate)

        for q in args:
            assert q.type == pytket.circuit.UnitType.qubit
            assert len(q.index) == 1

        for p in op.params:
            assert isinstance(p, float)

        name = op.gate.name
        # TKET normalizes in units of pi
        params = [3.141592653 * p for p in op.params]
        qubits = [q.index[0] for q in args]

        assert name in spec["gates"]
        assert len(params) == spec["gates"][name]["params"]
        assert len(qubits) == spec["gates"][name]["qubits"]
        for q in qubits:
            nqubits = max(q+1, nqubits)    

        ops.append((name, params, qubits, cbit, cond))

    assert nqubits > 0
    assert nqubits <= spec["qubits"]

    return ops, nqubits

def ops_to_qiskit(spec: dict, ops: list, nqubits: int, ncbits: int, scale: float, expval: str | None) -> tuple[qiskit.QuantumCircuit, qiskit.circuit.ParameterVector]:
    assert ncbits <= nqubits

    q = qiskit.QuantumRegister(nqubits)
    # ...sigh... qiskit why
    c = [qiskit.ClassicalRegister(1) for _ in range(ncbits)]
    r = qiskit.ClassicalRegister(nqubits)
    circ = qiskit.QuantumCircuit(*[q] + c + [r])

    # The ugliest hack of all time :)
    # Initializes the classical register in a (hopefully) batchable way
    if ncbits > 0:
        pv = qiskit.circuit.ParameterVector('cv', length=ncbits)
        for i in range(ncbits):
            circ.rx(pv[i], q[i])
            circ.measure(q[i], c[i])
            circ.reset(q[i])
            #circ.x(q[i]).c_if(c[i], 1)
    else:
        pv = None

    # Initialization errors are injected as X gate errors at the beginning
    for i in range(nqubits):
        circ.append(qiskit_aer.noise.pauli_error([('I', 1.0 - spec["spam"]["prepare"]), ('X', spec["spam"]["prepare"])]), [q[i]])

    for name, params, qubits, cbit, cond in ops:
        unitary = get_unitary(spec, name, params)
        inst = circ.unitary(unitary, [q[i] for i in qubits][::-1])
        if cbit is not None:
            inst.c_if(c[cbit], cond)

        noise = get_noise(spec, name, params, scale)
        if noise is not None:
            # If there is noise on this gate, add the corresponding operation
            inst = circ.append(qiskit_aer.noise.pauli_error(list(zip(*noise))), [q[i] for i in qubits][::-1])
            if cbit is not None:
                inst.c_if(c[cbit], cond)

    if expval is not None:
        qiskit_aer.library.save_expectation_value(circ, qiskit.quantum_info.Operator.from_label(expval), q, pershot=True)
    
    circ.measure(q, r)

    return circ, pv

#Add measurement errors to a set of noiseless samples
def inject_measurement_errors(spec: dict, noiseless_bits: np.ndarray, scale: float) -> np.ndarray:
    # The probability of a measurement error depends on the noiseless value
    probs = noiseless_bits * spec["spam"]["measure"]["one"] * scale + (1 - noiseless_bits) * spec["spam"]["measure"]["zero"] * scale
    # Flip samples at random with these probabilities
    flips = np.random.random(size=probs.shape) < probs
    noisy_bits = noiseless_bits ^ flips
    return noisy_bits

if __name__ == "__main__":
    # Read a circuit from disk
    circuit = pytket.qasm.circuit_from_qasm(args.CIRCUIT)
    spec = json.load(open(args.SPEC, "r"))
    shots = args.SHOTS
    if args.bits is not None:
        cbits = np.load(args.bits)
        ncbits = cbits.shape[1]
    else:
        cbits = np.array([])
        ncbits = 0
    out_file = pathlib.Path(args.output) if args.output is not None else pathlib.Path(args.CIRCUIT).with_suffix(".out.npy")

    # Get the operations for the circuit
    ops, nqubits = tket_to_ops(spec, circuit)
    circ, pv = ops_to_qiskit(spec, ops, nqubits, ncbits, args.scale, args.expval)

    # Build an AerSimulator
    if args.gpu is not None:
        backend = qiskit_aer.AerSimulator(
            method = "statevector",
            device = "GPU",
            cuStateVec_enable=True,
            # Blocked on: qiskit-aer#2244
            #target_gpus = args.gpu,
            precision = "single",
            batched_shots_gpu = True,
            batched_shots_gpu_max_qubits = 24,
            runtime_parameter_bind_enable = True,
            fusion_threshold = 30
        )
    else:
        backend = qiskit_aer.AerSimulator(
            method = "statevector",
            device = "CPU",
            precision = "single",
            runtime_parameter_bind_enable = False,
            max_parallel_experiments = 0,
            fusion_threshold = 30
        )

    circs = 1 if ncbits == 0 else cbits.shape[0]
    shot_outputs = np.zeros((circs, shots, nqubits), dtype=bool)
    actual_cbits = np.zeros((circs, ncbits))
    expval_outputs = np.zeros((circs, shots), dtype=float)
    for shot in tqdm.trange(shots, position=0, leave=False, dynamic_ncols=True, desc="simulation running"):
        results = []
        if ncbits > 0:
            for i in tqdm.trange(0, cbits.shape[0], args.batchsize, dynamic_ncols=True, position=1, leave=False):
                job = backend.run(circ, parameter_binds=[{ pv[j]: np.pi * cbits[i:i+args.batchsize, j] for j in range(cbits.shape[1]) }], shots=1)
                for result in job.result().results:
                    if args.expval is not None:
                        results.append((result.data.counts, result.data.expectation_value, result.header))
                    else:
                        results.append((result.data.counts, None, result.header))
                del job
        else:
            job = backend.run(circ, shots=1)
            for result in job.result().results:
                if args.expval is not None:
                    results.append((result.data.counts, result.data.expectation_value, result.header))
                else:
                    results.append((result.data.counts, None, result.header))
            del job
        
        
        for idx, (counts, expval, header) in enumerate(results):
            counts = qiskit.result.Counts(counts, memory_slots=header.memory_slots, creg_sizes=header.creg_sizes)
            shot_outputs[idx, shot, :] = np.array([bool(int(i)) for i in next(iter(counts))[:nqubits][::-1]])
            if ncbits > 0:
                actual_cbits[idx, :] = np.array([bool(int(i)) for i in next(iter(counts))[nqubits:][::-1].split()])
            if args.expval is not None:
                expval_outputs[idx, shot] = expval[0]
    shot_outputs = inject_measurement_errors(spec, shot_outputs, args.scale)

    np.save(out_file, shot_outputs)
    print(f"wrote shot values of shape {shot_outputs.shape} to `{out_file}`")
    if args.expval:
        np.save(out_file.with_suffix('.exp.npy'), expval_outputs)
        print(f"wrote expectation values of shape {expval_outputs.shape} to `{out_file.with_suffix('.exp.npy')}`")
