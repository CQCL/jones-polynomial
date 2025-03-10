import argparse
import rich_argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=rich_argparse.RichHelpFormatter)
    parser.add_argument("SPEC", help="machine specification file")
    parser.add_argument("ANSATZE", help="quantum circuit ansatze file")
    parser.add_argument("OUTPUT", help="folder for output files")
    parser.add_argument("-l", "--lr", help="learning rate for optimization", type=float, default=0.01)
    parser.add_argument("-s", "--steps", help="number of steps of gradient descent", type=int, default=4000)
    parser.add_argument("-p", "--prompt", help="prompt for continue/exit/retry after each optimization", action='store_true')
    parser.add_argument("--power", help="train just this specific power rather than all of them", type=int)
    args = parser.parse_args()

import torch
import string
import itertools
import pytket
import pytket.qasm
import cmath
import json
import tqdm
import pathlib
import numpy as np

paulis = {
    "I": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.complex128),
    "X": torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128),
    "Y": torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=torch.complex128),
    "Z": torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex128),
}

def kron(*args):
    if len(args) == 1:
        return args[0]
    else:
        return torch.kron(args[0], kron(*args[1:]))

def assemble_tensor(mat: list[list[torch.Tensor | complex]], dtype=None) -> torch.Tensor:
    return torch.stack([
        torch.stack([
            (entry if dtype is None else entry.to(dtype=dtype)) 
            if isinstance(entry, torch.Tensor) else torch.tensor(entry, dtype=dtype)
            for entry in row
        ], dim=0)
        for row in mat
    ], dim=0)

def to_qubits(mat: torch.Tensor, qubits: list[int], total: int) -> torch.Tensor:
    mat = torch.kron(mat, torch.eye(2**(total - len(qubits))))
    mat = mat.reshape([2]*(2*total))
    perm = [None]*(2*total)
    for i, qb in enumerate(qubits):
        perm[qb] = i
        perm[total + qb] = total + i
    j = len(qubits)
    for k in range(total):
        if perm[k] is None:
            perm[k] = j
            perm[total + k] = total + j
            j += 1
    mat = torch.permute(mat, perm)
    return mat.reshape(2**total, 2**total)

def get_unitary(spec: dict, gate: str, params: list[torch.Tensor], qubits: list[int], total: int) -> torch.Tensor:
    unitary = spec["gates"][gate]["unitary"]

    if isinstance(unitary, dict):
        # Otherwise it is a dictionary with a code attribute
        params_dict = { l: p for l, p in zip(string.ascii_lowercase, params) }
        # Evaluate the code with the parameters as locals to get the matrix
        val = eval(unitary["code"], torch.__dict__.copy(), params_dict)
        u = assemble_tensor(val, dtype=torch.complex128)
        u = to_qubits(u, qubits, total)
    elif isinstance(unitary, list):
        u = assemble_tensor(val, dtype=torch.complex128)
        u = to_qubits(u, qubits, total)
    else:
        raise ValueError()
    
    return u

# Get the superoperator matrix corresponding to a native gate with noise
def get_superop(spec: dict, gate: str, params: list[torch.Tensor], qubits: list[int], total: int, noiseless: bool) -> torch.Tensor: 
    q = spec["gates"][gate]["qubits"]

    u = get_unitary(spec, gate, params, qubits, total)
    op_u = torch.kron(u, u.conj())

    if noiseless or "error" not in spec["gates"][gate]:
        return op_u
    
    rate = spec["gates"][gate]["error"]["rate"]
    channel = spec["gates"][gate]["error"]["channel"]

    if isinstance(rate, float):
        rate = torch.tensor(rate)
    elif isinstance(rate, dict):
        # Rate may depend on the parameters, in which case
        # we will have a dictionary with code to eval
        params_dict = { l: p for l, p in zip(string.ascii_lowercase, params) }
        rate = eval(rate["code"], torch.__dict__.copy(), params_dict)
    else:
        raise ValueError()

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

    # Create a superoperator matrix corresponding to the noise channel
    op = (1.0 - rate) * torch.eye((2**total) * (2**total))
    for p, r in channel.items():
        pauli = kron(*(paulis[pm] for pm in p))
        pauli = to_qubits(pauli, qubits, total)
        op = op + (r * rate) * torch.kron(pauli, pauli.conj())
    op_u = op @ op_u
    
    return op_u

def tket_to_ops(spec: dict, circ: pytket.circuit.Circuit) -> tuple[list[tuple[str, list[int]]], int, int]:
    assert len(circ.q_registers) == 1
    assert len(circ.c_registers) == 0

    ops = []
    nqubits = 0
    nparams = 0
    for command in circ.get_commands():
        op, args = command.op, command.args

        assert isinstance(op, pytket.circuit.CustomGate)

        for q in args:
            assert q.type == pytket.circuit.UnitType.qubit
            assert len(q.index) == 1

        for p in op.params:
            assert isinstance(p, float)

        name = op.gate.name
        qubits = [q.index[0] for q in args]

        assert name in spec["gates"]
        assert len(op.params) == spec["gates"][name]["params"]
        assert len(qubits) == spec["gates"][name]["qubits"]
        for q in qubits:
            nqubits = max(q+1, nqubits)    
        nparams += len(op.params)

        ops.append((name, qubits))

    assert nqubits > 0
    assert nqubits <= spec["qubits"]

    return ops, nqubits, nparams

def ops_to_superop(spec: dict, ops: list[tuple[str, list[int]]], params: torch.Tensor, total: int, noiseless: bool) -> torch.Tensor:
    mat = torch.eye(2**total * 2**total, dtype=torch.complex128)
    pidx = 0
    for gate, qubits in ops:
        nparams = spec["gates"][gate]["params"]
        mat = get_superop(spec, gate, [params[pidx+i] for i in range(nparams)], qubits, total, noiseless) @ mat
        pidx += nparams
    return mat

def ops_to_unitary(spec: dict, ops: list[tuple[str, list[int]]], params: torch.Tensor, total: int) -> torch.Tensor:
    mat = torch.eye(2**total, dtype=torch.complex128)
    pidx = 0
    for gate, qubits in ops:
        nparams = spec["gates"][gate]["params"]
        mat = get_unitary(spec, gate, [params[pidx+i] for i in range(nparams)], qubits, total) @ mat
        pidx += nparams
    return mat

def process_fidelity(superop: torch.Tensor, unitary: torch.Tensor, indices: list[int], total: int) -> torch.Tensor:
    uop = torch.kron(unitary.T.conj(), unitary.T)
    #idx2 = [j*2**total + i for j in indices for i in indices]
    #superop = superop[idx2, :][:, idx2]
    #uop = uop[idx2, :][:, idx2]
    return torch.real(torch.trace(uop @ superop) / len(indices) ** 2)

def mse_unitary(superop: torch.Tensor, unitary: torch.Tensor, indices: list[int], total: int) -> torch.Tensor:
    superop = superop[indices, :][:, indices]
    unitary = unitary[indices, :][:, indices]
    diff = unitary - superop
    return torch.sum(diff.real**2 + diff.imag**2)

def make_target(eigenphases: torch.Tensor, power: int) -> torch.Tensor:
    A = cmath.exp((3/5)*1j*torch.pi)
    Ainv = cmath.exp((-3/5)*1j*torch.pi)
    delta = (cmath.sqrt(5) + 1) / 2
    a = 1 / delta
    b = cmath.sqrt(1 - a ** 2)
    target = assemble_tensor([
        [torch.exp(1j*torch.pi*eigenphases[0]), 0, 0, 0, 0, 0, 0, 0],
        [0, torch.exp(1j*torch.pi*eigenphases[1]), 0, 0, 0, 0, 0, 0],
        [0, 0, Ainv*delta + A, 0, 0, 0, 0, 0],
        [0, 0, 0, A, 0, 0, 0, 0],
        [0, 0, 0, 0, torch.exp(1j*torch.pi*eigenphases[2]), 0, 0, 0],
        [0, 0, 0, 0, 0, Ainv*a + A, 0, Ainv*b],
        [0, 0, 0, 0, 0, 0, A, 0],
        [0, 0, 0, 0, 0, Ainv*b, 0, Ainv*delta*b**2 + A]
    ], dtype=torch.complex128)
    target = torch.linalg.matrix_power(target, torch.tensor(power))
    return target, [0, 1, 2, 3, 4, 5, 6, 7]

def fit_ansatze(spec: dict, ops: list, nparams: int, power: int, method: str):
    best_loss = float('inf')
    best_infidelity = float('inf')
    best_params = None

    global_phase = torch.nn.Parameter(torch.zeros((), dtype=torch.float64, requires_grad=True), requires_grad=True)
    eigenphase = torch.nn.Parameter(torch.rand(3, dtype=torch.float64, requires_grad=True), requires_grad=True)
    params = torch.nn.Parameter(torch.rand(nparams, dtype=torch.float64, requires_grad=True), requires_grad=True)
    optim = torch.optim.Adam([eigenphase, params], lr=args.lr)
    pbar = tqdm.trange(args.steps)
    for _ in pbar:
        optim.zero_grad()

        def closure():
            target, indices = make_target(eigenphase, power)
            unitary = torch.exp(1j * global_phase) * ops_to_unitary(spec, ops, params, 3)
            loss = mse_unitary(unitary, target, indices, 3)
            return loss
        
        if method == 'adam':
            loss = closure()
            loss.backward()
            optim.step()
        else:
            loss = optim.step(closure)

        if loss.item() < best_loss:
            target, indices = make_target(eigenphase, power)
            with torch.no_grad():
                superop_noisy = ops_to_superop(spec, ops, params % (2*torch.pi), 3, False)
                infidelity = 1.0 - process_fidelity(superop_noisy, target, indices, 3)

            best_loss = loss.item()
            best_infidelity = infidelity.item()
            best_params = (global_phase.item(), eigenphase[0].item(), params.detach().numpy() % (2*np.pi))

        pbar.set_description(f"best loss = {best_loss:g}, infidelity = {best_infidelity:0>.5f}")
    
    return best_loss, best_infidelity, best_params

def ops_to_qasm(spec: dict, ops: list[tuple[str, list[int]]], params: np.ndarray) -> str:
    lines = ['OPENQASM 2.0;', '']
    for gate in spec["gates"]:
        nparams = spec["gates"][gate]["params"] 
        nqubits = spec["gates"][gate]["qubits"] 
        sparams = '(' + ', '.join(f'p{i}' for i in range(nparams)) + ')' if nparams > 0 else ""
        squbits = ", ".join(f'q{i}' for i in range(nqubits))
        lines.append(f'opaque {gate}{sparams} {squbits};')
    lines.extend(['', 'qreg q[3];', ''])

    pidx = 0
    for gate, qubits in ops:
        nparams = spec["gates"][gate]["params"] 
        nqubits = spec["gates"][gate]["qubits"] 
        sparams = '(' + ', '.join(str(params[pidx+i]) for i in range(nparams)) + ')' if nparams > 0 else ""
        squbits = ', '.join(f'q[{i}]' for i in qubits)
        lines.append(f'{gate}{sparams} {squbits};')
        pidx += nparams
    lines.append('')
    
    return '\n'.join(lines)

if __name__ == "__main__":
    circuit = pytket.qasm.circuit_from_qasm(args.ANSATZE)
    spec = json.load(open(args.SPEC, "r"))

    ops, nqubits, nparams = tket_to_ops(spec, circuit)
    assert nqubits == 3

    powers = [1, 2, 3, 4, 5, 6, 7, 8, 9] if args.power is None else [args.power]
    for power in powers:
        print(f"Training power = {power}:")
        save = True
        while True:
            try:
                loss, inf, params = fit_ansatze(spec, ops, nparams, power, args.method)
                print(f"Found process infidelity = {inf:.5g}")
            except KeyboardInterrupt:
                pass
            if args.prompt:
                while True:
                    response = input("Do you want to (s)ave, (r)etry, or (c)ontinue without saving? ")
                    if response in ('s', 'r', 'c'):
                        break
                    else:
                        print("Error: response must be 's', 'c', or 'r'.")
                if response == 's':
                    break
                elif response == 'r':
                    continue
                elif response == 'c':
                    save = False
                    break
        
        if save == False:
            continue
        
        output_path = pathlib.Path(args.OUTPUT) / f'{pathlib.Path(args.SPEC).stem}_{pathlib.Path(args.ANSATZE).stem}_p{power}'
        print(f"Saving: {output_path}")
        
        unitary = ops_to_unitary(spec, ops, torch.tensor(params[2]), 3).numpy().tolist()

        with open(output_path.with_suffix('.json'), 'w') as file:
            json.dump({
                'power': power,
                'loss': loss,
                'process_infidelity': inf,
                'global_phase': params[0],
                'zero_eigenphase': params[1],
                'params': params[2].tolist(),
                'ops': ops,
                'unitary': [[(c.real, c.imag) for c in row] for row in unitary]
            }, file, indent=4)

        with open(output_path.with_suffix('.qasm'), 'w') as file:
            file.write(ops_to_qasm(spec, ops, params[2]))
