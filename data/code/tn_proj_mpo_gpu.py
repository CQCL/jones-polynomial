import copy
from tqdm import tqdm
import pickle
import numpy as np
import torch
import quimb.tensor as qtn
import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description="Compute Jones Polynomials from MPO"
    )
    parser.add_argument(
        "filename", type=str, help="Name of the file with braids"
    )
    parser.add_argument(
        "instance_min", type=int, help="Smallest index for the instance"
    )
    parser.add_argument(
        "instance_max", type=int, help="Largest index for the instance"
    )
    parser.add_argument(
        "bond_dimension", type=int, help="Value for the bond dimension"
    )
    parser.add_argument(
        "-r", "--record-dims", action="store_true", help="Record the bond dimensions throughout the computation"
    )

    args = parser.parse_args()

    filename = args.filename
    initial_instance = args.instance_min
    final_instance = args.instance_max
    max_bond = args.bond_dimension

    # If max_bond = 0, let quimb pick the required bond dim
    if max_bond == 0:
        max_bond = None

    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    with open(filename, "rb") as file:
        loaded = pickle.load(file)

    for instance in range(initial_instance, final_instance + 1):
        num_strands = loaded[instance]["strands"]
        crossing_list = loaded[instance]["crossing_list"]
        
        if args.record_dims:
            mpo_est, bond_sizes = compute_weighted_trace_from_mpo(
                num_strands,
                crossing_list,
                prog_bar=True,
                max_bond=max_bond,
                return_chi=True,
            )
            loaded[instance][f"mpo-{args.bond_dimension}"] = {
                'estimate': mpo_est,
                'bond_sizes': bond_sizes
            }
            print(f"MPO (chi={args.bond_dimension}): {mpo_est}, max chi = {max(max(b) for b in bond_sizes)}")
        else:
            mpo_est = compute_weighted_trace_from_mpo(
                num_strands,
                crossing_list,
                prog_bar=True,
                max_bond=max_bond,
                return_chi=False,
            )
            loaded[instance][f"mpo-{args.bond_dimension}"] = mpo_est
            print(f"MPO (chi={args.bond_dimension}): {mpo_est}")

    output_filename = (
        f"mpo_d{args.bond_dimension}_{initial_instance}-{final_instance}.pkl"
    )
    with open(output_filename, "wb") as output_file:
        pickle.dump(loaded[initial_instance:final_instance + 1], output_file)
    print(f"Saved instances to {output_filename}")


def to_torch_array(x):
    return torch.tensor(x, dtype=torch.complex128, device="cuda:0")


def jones_factor(
    num_strands,
    crossing_list,
):
    """
    Factor to convert from MPO expectation value -> Jones polynomial

    Parameters
    ----------
    num_strands : int
    crossing_list : iterable[tuple]
        Specify the crossings in order, each entry is a 2-tuple (i,s),
        the elements have meanings:
            i : a strand index, the crossing is added between i and i+1
            s : the sign of the crossing, +1 or -1

    Returns
    -------
    float
    """
    writhe = sum([val[1] for val in crossing_list])
    n_qubits = num_strands + 1
    phi = (1 + np.sqrt(5)) / 2.0
    return (-np.exp(-1j * 3 * np.pi / 5)) ** (3 * writhe) \
        * phi ** (n_qubits - 2)


def make_braid_generator():
    """
    Make the dense 2d array of the braid generator
    """
    phi = (1 + np.sqrt(5)) / 2.0

    U = np.zeros((2**3, 2**3), dtype=np.complex128)
    # non-fibonacci subspace
    U[0, 0] = 0
    U[1, 1] = 0
    U[4, 4] = 0
    # fibonacci subspace
    U[2, 2] = np.exp(-1j * 4 * np.pi / 5)
    U[3, 3] = np.exp(1j * 3 * np.pi / 5)
    U[5, 5] = np.exp(1j * 4 * np.pi / 5) / phi
    U[6, 6] = np.exp(1j * 3 * np.pi / 5)
    U[7, 7] = -1 / phi
    U[7, 5] = np.exp(-1j * 3 * np.pi / 5) / np.sqrt(phi)
    U[5, 7] = np.exp(-1j * 3 * np.pi / 5) / np.sqrt(phi)

    return U


def make_projectors():
    """
    Make the dense 2d array of the three projectors
    """
    phi = (1 + np.sqrt(5)) / 2.0

    proj_boundary_0 = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    proj_bulk = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    proj_boundary_1 = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    ) + phi * np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    return (
        proj_boundary_0,
        proj_bulk,
        proj_boundary_1,
    )


def make_braid_mpo(
    num_strands,
    crossing_list,
    max_bond=None,
    prog_bar=False,
    debug_draw=False,
    return_chi=False,
):
    """
    Make the MPO of a braid, does not include the projectors

    Parameters
    ----------
    num_strands : int
    crossing_list : iterable[tuple]
        Specify the crossings in order, each entry is a 2-tuple (i,s),
        the elements have meanings:
            i : a strand index, the crossing is added between i and i+1
            s : the sign of the crossing, +1 or -1
    max_bond : (optional) int
        Passed to the TN contraction when the MPO corresponding to each
        crossing are contracted together
    prog_bar : (optional) bool
        If set to True will display progress bar of building MPO
    debug_draw : (optional) bool
        If set to True will display drawings helpful for debugging
    return_chi : (optional) bool
        If set to True will return the bond dimensions of the MPO throughout the evolution 

    Returns
    -------
    quimb.tensor.MatrixProductOperator
    list[list[int]] (if return_chi = True)
    """
    num_qubits = num_strands + 1

    U = make_braid_generator()
    Udag = U.T.conj()
    U = to_torch_array(U)
    Udag = to_torch_array(Udag)

    next_crossing = crossing_list[0]
    if next_crossing[1] == 1:
        tmp = U
    elif next_crossing[1] == -1:
        tmp = Udag
    else:
        raise ValueError(
            f"Cannot interpret sign of crossing: {next_crossing[1]}. \
                Should have value +1 or -1."
        )

    braid_mpo = qtn.MatrixProductOperator.from_dense(
        tmp,
        L=num_qubits,
        sites=[
            next_crossing[0],
            next_crossing[0] + 1,
            next_crossing[0] + 2,
        ],
    )
    braid_mpo = braid_mpo.fill_empty_sites()
    if debug_draw:
        braid_mpo.draw(color=[f"I{i}" for i in range(num_qubits)])

    if prog_bar:
        _iterator = tqdm(range(1, len(crossing_list)))
    else:
        _iterator = range(1, len(crossing_list))

    bond_sizes = [braid_mpo.bond_sizes()]
    for idx in _iterator:
        next_crossing = crossing_list[idx]
        if next_crossing[1] == 1:
            tmp = U
        elif next_crossing[1] == -1:
            tmp = Udag
        else:
            raise ValueError(
                f"Cannot interpret sign of crossing: {next_crossing[1]}. \
                    Should have value +1 or -1."
            )

        next_layer = qtn.MatrixProductOperator.from_dense(
            tmp,
            L=num_qubits,
            sites=[
                next_crossing[0],
                next_crossing[0] + 1,
                next_crossing[0] + 2,
            ],
        )
        next_layer = next_layer.fill_empty_sites()
        if debug_draw:
            next_layer.draw(color=[f"I{i}" for i in range(num_qubits)])

        braid_mpo = next_layer.apply(
            braid_mpo,
            compress=True,
            max_bond=max_bond,
        )
        bond_sizes.append(braid_mpo.bond_sizes())

    if not return_chi:
        return braid_mpo
    else:
        return braid_mpo, bond_sizes


def add_projectors_to_braid_mpo(
    num_qubits,
    braid_mpo,
    max_bond=None,
    prog_bar=False,
    debug_draw=False,
    return_chi=False,
):
    """
    Add the projectors onto the end of a braid MPO

    Parameters
    ----------
    num_qubits : int
    braid_mpo : quimb.tensor.MatrixProductOperator
    max_bond : (optional) int
        Passed to the TN contraction when the MPO corresponding to each
        crossing are contracted together
    prog_bar : (optional) bool
        If set to True will display progress bar of building MPO
    debug_draw : (optional) bool
        If set to True will display drawings helpful for debugging
    return_chi : (optional) bool
        If set to True will return the bond dimensions of the MPO throughout the evolution 

    Returns
    -------
    quimb.tensor.MatrixProductOperator
    list[list[int]] (if return_chi = True)
    """
    proj_boundary_0, proj_bulk, proj_boundary_1 = make_projectors()
    proj_boundary_0 = to_torch_array(proj_boundary_0)
    proj_bulk = to_torch_array(proj_bulk)
    proj_boundary_1 = to_torch_array(proj_boundary_1)

    proj_mpo = qtn.MatrixProductOperator.from_dense(
        proj_boundary_0,
        L=num_qubits,
        sites=[0, 1],
    )
    if debug_draw:
        proj_mpo.draw(color=[f"I{i}" for i in range(num_qubits)])
    proj_mpo = proj_mpo.fill_empty_sites()
    braid_mpo = proj_mpo.apply(
        braid_mpo,
        compress=True,
        max_bond=max_bond,
    )
    bond_sizes = [braid_mpo.bond_sizes()]

    if prog_bar:
        _iterator = tqdm(range(1, num_qubits - 2))
    else:
        _iterator = range(1, num_qubits - 2)

    for idx in _iterator:
        proj_mpo = qtn.MatrixProductOperator.from_dense(
            proj_bulk,
            L=num_qubits,
            sites=[idx, idx + 1],
        )
        if debug_draw:
            proj_mpo.draw(color=[f"I{i}" for i in range(num_qubits)])
        proj_mpo = proj_mpo.fill_empty_sites()
        braid_mpo = proj_mpo.apply(
            braid_mpo,
            compress=True,
            max_bond=max_bond,
        )
        bond_sizes.append(braid_mpo.bond_sizes())

    proj_mpo = qtn.MatrixProductOperator.from_dense(
        proj_boundary_1,
        L=num_qubits,
        sites=[num_qubits - 2, num_qubits - 1],
    )
    if debug_draw:
        proj_mpo.draw(color=[f"I{i}" for i in range(num_qubits)])
    proj_mpo = proj_mpo.fill_empty_sites()
    braid_mpo = proj_mpo.apply(
        braid_mpo,
        compress=True,
        max_bond=max_bond,
    )
    bond_sizes.append(braid_mpo.bond_sizes())

    if not return_chi:
        return braid_mpo
    else:
        return braid_mpo, bond_sizes


def make_markov_closure_mpo(
    num_strands,
    crossing_list,
    max_bond=None,
    prog_bar=False,
    debug_draw=False,
    return_chi=False,
):
    """
    Make the MPO of a braid including the projectors

    Parameters
    ----------
    num_strands : int
    crossing_list : iterable[tuple]
        Specify the crossings in order, each entry is a 2-tuple (i,s),
        the elements have meanings:
            i : a strand index, the crossing is added between i and i+1
            s : the sign of the crossing, +1 or -1
    max_bond : (optional) int
        Passed to the TN contraction when the MPO corresponding to each
        crossing are contracted together
    prog_bar : (optional) bool
        If set to True will display progress bar of building MPO
    debug_draw : (optional) bool
        If set to True will display drawings helpful for debugging
    return_chi : (optional) bool
        If set to True will return the bond dimensions of the MPO throughout the evolution 

    Returns
    -------
    quimb.tensor.MatrixProductOperator
    list[list[int]] (if return_chi = True)
    """

    if return_chi:
        braid_mpo, bond_sizes_braid = make_braid_mpo(
            num_strands,
            crossing_list,
            max_bond=max_bond,
            prog_bar=prog_bar,
            debug_draw=debug_draw,
            return_chi=True,
        )
        braid_mpo, bond_sizes_proj = add_projectors_to_braid_mpo(
            num_strands + 1, 
            braid_mpo, 
            max_bond=max_bond, 
            debug_draw=debug_draw, 
            return_chi=True,
        )
        bond_sizes = bond_sizes_braid + bond_sizes_proj
        return braid_mpo, bond_sizes
    else:
        braid_mpo = make_braid_mpo(
            num_strands,
            crossing_list,
            max_bond=max_bond,
            prog_bar=prog_bar,
            debug_draw=debug_draw,
            return_chi=False,
        )
        return add_projectors_to_braid_mpo(
            num_strands + 1, 
            braid_mpo, 
            max_bond=max_bond, 
            debug_draw=debug_draw, 
            return_chi=False,
        )


def compute_weighted_trace_from_mpo(
    num_strands,
    crossing_list,
    max_bond=None,
    prog_bar=False,
    debug_draw=False,
    return_chi=False,
):
    """
    Make and trace the MPO of a braid including the projectors

    Parameters
    ----------
    num_strands : int
    crossing_list : iterable[tuple]
        Specify the crossings in order, each entry is a 2-tuple (i,s),
        the elements have meanings:
            i : a strand index, the crossing is added between i and i+1
            s : the sign of the crossing, +1 or -1
    max_bond : (optional) int
        Passed to the TN contraction when the MPO corresponding to each
        crossing are contracted together
    prog_bar : (optional) bool
        If set to True will display progress bar of building MPO
    debug_draw : (optional) bool
        If set to True will display drawings helpful for debugging
    return_chi : (optional) bool
        If set to True will return the bond dimensions of the MPO throughout the evolution 

    Returns
    -------
    numpy.complex128 (if using CPU backend)
        OR numpy.complex64 (if using GPU backend)
    list[list[int]] (if return_chi = True)
    """
    if not return_chi:
        braid_mpo_with_projectors = make_markov_closure_mpo(
            num_strands,
            crossing_list,
            max_bond=max_bond,
            prog_bar=prog_bar,
            debug_draw=debug_draw,
            return_chi=False
        )
    else:
        braid_mpo_with_projectors, bond_sizes = make_markov_closure_mpo(
            num_strands,
            crossing_list,
            max_bond=max_bond,
            prog_bar=prog_bar,
            debug_draw=debug_draw,
            return_chi=True
        )

    _trace = braid_mpo_with_projectors.trace()
    _trace = _trace.cpu().numpy()

    phi = (1 + np.sqrt(5)) / 2.0
    if not return_chi:
        return _trace / phi ** (num_strands)
    else:
        return _trace / phi ** (num_strands), bond_sizes


def make_braid_unitary_dense(num_strands, crossing_list):
    """
    for testing
    """
    num_qubits = num_strands + 1
    U = make_braid_generator()
    Udag = U.T.conj()
    _crossing_list = copy.copy(crossing_list)

    iden = np.array(
        [
            [1, 0],
            [0, 1],
        ]
    )

    next_crossing = _crossing_list.pop(0)
    if next_crossing[1] == 1:
        tmp = U
    elif next_crossing[1] == -1:
        tmp = Udag
    else:
        raise ValueError(
            f"Cannot interpret sign of crossing: {next_crossing[1]}. \
                Should have value +1 or -1."
        )

    if next_crossing[0] == 0:
        braid_dense = tmp
    else:
        braid_dense = iden
        for _ in range(1, next_crossing[0]):
            braid_dense = np.kron(braid_dense, iden)
        braid_dense = np.kron(braid_dense, tmp)
    for _ in range(next_crossing[0] + 2, num_qubits - 1):
        braid_dense = np.kron(braid_dense, iden)

    while len(_crossing_list) > 0:
        next_crossing = _crossing_list.pop(0)
        if next_crossing[1] == 1:
            tmp = U
        elif next_crossing[1] == -1:
            tmp = Udag
        else:
            raise ValueError(
                f"Cannot interpret sign of crossing: {next_crossing[1]}. \
                    Should have value +1 or -1."
            )

        if next_crossing[0] == 0:
            next_braid_dense = tmp
        else:
            next_braid_dense = iden
            for _ in range(1, next_crossing[0]):
                next_braid_dense = np.kron(next_braid_dense, iden)
            next_braid_dense = np.kron(next_braid_dense, tmp)
        for _ in range(next_crossing[0] + 2, num_qubits - 1):
            next_braid_dense = np.kron(next_braid_dense, iden)

        braid_dense = next_braid_dense @ braid_dense

    return braid_dense


def gen_fibo_strings(length):
    """
    for testing, exhaustive
    """
    _strs = []
    for int_val in range(2**length):
        _bin_str = bin(int_val)[2:]
        if len(_bin_str) > length:
            raise ValueError("Got binary string longer than expected.")
        _bin_str = "0" * (length - len(_bin_str)) + _bin_str
        _bin_tuple = [int(x) for x in list(_bin_str)]
        if _bin_tuple[0] == 0:
            _discard = False
            for idx in range(len(_bin_tuple) - 1):
                if (_bin_tuple[idx] == 0) and (_bin_tuple[idx + 1] == 0):
                    _discard = True
                    break
            if not _discard:
                _strs.append(_bin_tuple)
    return _strs


def fibo_number(n):
    """
    for testing
    """
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibo_number(n - 1) + fibo_number(n - 2)


def compute_weighted_trace_from_dense(num_strands, crossing_list):
    """
    for testing
    """
    phi = (1 + np.sqrt(5)) / 2.0
    num_qubits = num_strands + 1
    braid_dense = make_braid_unitary_dense(num_strands, crossing_list)

    _weighted_trace = 0.0
    fibo_strs = gen_fibo_strings(num_qubits)
    for bin_tuple in fibo_strs:
        int_val = int("".join([str(x) for x in bin_tuple]), base=2)
        weighting = (phi ** bin_tuple[-1]) / (phi ** (num_qubits - 1))
        _weighted_trace += weighting * braid_dense[int_val, int_val]

    return _weighted_trace


if __name__ == "__main__":
    main()
