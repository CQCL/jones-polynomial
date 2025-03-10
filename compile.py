import argparse
import rich_argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=rich_argparse.RichHelpFormatter)
    parser.add_argument("SPEC", help="machine specification file")
    parser.add_argument("ANSATZE", help="quantum circuit ansatze file")
    parser.add_argument("BRAID", help="braid json files to compile", nargs="+")
    parser.add_argument("-c", "--circuit-dir", help="directory for finetuned braid generators", default="circuits")
    parser.add_argument("-o", "--out-dir", help="directory to put output files", default=".")
    args = parser.parse_args()

from common import Braid
import minimize
import json
import pathlib
import string
import cmath

def eval_param(param, args: list[float]) -> float:
    if isinstance(param, float):
        # If it's a float, we are already done
        return param
    elif isinstance(param, dict):
        # Otherwise it is a dictionary with a code attribute
        args_dict = { l: p for l, p in zip(string.ascii_lowercase, args) }
        # Evaluate the code with the parameters as locals to get the value
        return eval(param["code"], cmath.__dict__.copy(), args_dict)
    else:
        raise ValueError()

def construct_vcat(spec: dict, qubits: int, dagger: bool) -> list[tuple[str, list[int], list[float], int, int]]:
    cnots = []
    if spec["connectivity"] == "all":
        for i in range(2, qubits):
            # Condition the cnot from qubit 1 to i by classical bit i-1
            # so cbits 1..n-1 are bits 2..n of s (cbit 0 is reserved for later)
            cnots.append((1, i, i-1, 1))
    elif spec["connectivity"] == "nearest":
        for i in range(2, qubits-1):
            cnots.append((i-1, i, None, None))

        for i in range(qubits-1, 1, -1):
            cnots.append((i-1, i, i-1, 0))
    else:
        raise ValueError("Only all-to-all and nearest-neighbour connectivities are supported")
    
    if dagger:
        cnots.reverse()

    ops = []
    for i, j, a, b in cnots:
        for name, qubits, params in spec["cnot_recipe"]:
            ops.append((name, [i if q == 0 else j for q in qubits], [eval_param(p, []) for p in params], a, b))
    
    return ops

def get_generator_power(spec: dict, folder: str, prefix: str, power: int, offset: int) -> tuple[float, float, list[tuple[str, list[int], list[float], int, int]]]:
    power %= 10
    if power == 0:
        return 0.0, []
    
    with open(pathlib.Path(folder) / (prefix + f'_p{power}.json')) as f:
        gspec = json.load(f)

    assert power == gspec["power"]
    
    ops = []
    pidx = 0
    for name, qubits in gspec["ops"]:
        nparams = spec["gates"][name]["params"]
        ops.append((
            name, [q + offset for q in qubits], gspec["params"][pidx:pidx+nparams], None, None
        ))
        pidx += nparams

    return (cmath.pi * power * gspec["zero_eigenphase"]) % (2*cmath.pi), gspec["process_infidelity"], ops

def get_braid_unitary(spec: dict, folder: str, prefix: str, braid: Braid) -> tuple[float, float, list[tuple[str, list[int], list[float], int, int]]]:
    ops = []
    eigenphase = 0.0
    fidelity = 1.0
    for run in minimize.find_consecutive(braid):
        power = braid[run[0]].sign * len(run)
        offset = braid[run[0]].pos - 1
        phase, infidelity, op = get_generator_power(spec, folder, prefix, power, offset)
        eigenphase += phase
        fidelity *= 1.0 - infidelity
        ops.extend(op)
    eigenphase %= (2*cmath.pi)

    return eigenphase, 1.0 - fidelity, ops

def construct_cfev(spec: dict, folder: str, prefix: str, braid: Braid) -> tuple[float, list[tuple[str, list[int], list[float], int, int]]]:
    ops = []
    nqubits = braid.strands + 1
    
    # Apply an H gate to qubit 1
    for name, qubits, params in spec["hadamard_recipe"]:
        assert qubits == [0]
        ops.append((name, [1], [eval_param(p, []) for p in params], None, None))

    # Apply an S^+ gate to qubit 1 conditioned on cbit 0
    for name, qubits, params in spec["rz_recipe"]:
        assert qubits == [0]
        ops.append((name, [1], [eval_param(p, [-cmath.pi/2.0]) for p in params], 0, 1))

    # Apply V_cat
    ops.extend(construct_vcat(spec, nqubits, False))

    # Apply the braid unitary
    eigenphase, infidelity, braid_ops = get_braid_unitary(spec, folder, prefix, braid)
    ops.extend(braid_ops)

    # Apply V_cat^+
    ops.extend(construct_vcat(spec, nqubits, True))

    # Apply Rz(eigenphase) to qubit 1
    for name, qubits, params in spec["rz_recipe"]:
        assert qubits == [0]
        ops.append((name, [1], [eval_param(p, [eigenphase]) for p in params], None, None))

    # Apply an H gate to qubit 1
    for name, qubits, params in spec["hadamard_recipe"]:
        assert qubits == [0]
        ops.append((name, [1], [eval_param(p, []) for p in params], None, None))
    
    return infidelity, ops

def ops_to_qasm(spec: dict, ops: list[tuple[str, list[int], list[float], int, int]], cbits: int, qubits: int) -> str:
    lines = ['OPENQASM 2.0;', '']
    for gate in spec["gates"]:
        nparams = spec["gates"][gate]["params"] 
        nqubits = spec["gates"][gate]["qubits"] 
        sparams = '(' + ', '.join(f'p{i}' for i in range(nparams)) + ')' if nparams > 0 else ""
        squbits = ", ".join(f'q{i}' for i in range(nqubits))
        lines.append(f'opaque {gate}{sparams} {squbits};')
    lines.extend(['', f'qreg q[{qubits}];', f'creg c[{cbits}];', ''])

    for gate, qubits, params, cbit, cond in ops:
        sparams = '(' + ', '.join(str(p) for p in params) + ')' if len(params) > 0 else ""
        squbits = ', '.join(f'q[{i}]' for i in qubits)
        if cbit is None:
            lines.append(f'{gate}{sparams} {squbits};')
        else:
            lines.append(f'if (c[{cbit}] == {cond}) {gate}{sparams} {squbits};')
    lines.append('')
    
    return '\n'.join(lines)

if __name__ == "__main__":
    folder = args.circuit_dir
    prefix = f'{pathlib.Path(args.SPEC).stem}_{pathlib.Path(args.ANSATZE).stem}'
    spec = json.load(open(args.SPEC, "r"))

    for fname in args.BRAID:
        with open(fname, "r") as f:
            bspec = json.load(f)
        braid = Braid.from_word(bspec["optimized"]["word"])
        infidelity, ops = construct_cfev(spec, folder, prefix, braid)
        tqcount = sum(len(qubits) > 1 for _, qubits, _, _, _ in ops)
        qasm = ops_to_qasm(spec, ops, braid.strands, braid.strands + 1)
        ofile = pathlib.Path(args.out_dir) / f'{pathlib.Path(fname).stem}_{prefix}.qasm'
        with open(ofile, "w") as f:
            f.write(qasm)
        print(f"wrote `{ofile}`, braid unitary infidelity = {infidelity*100.0:.2f}%, multi-qubit gate count = {tqcount}")
