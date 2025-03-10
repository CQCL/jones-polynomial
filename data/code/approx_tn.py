import copy
from tqdm import tqdm

import numpy as np

import quimb.tensor as qtn


# if set to true will run tests at start of __main__
RUN_TESTS = True

# if set to true will use torch cuda as a GPU backend
USE_GPU = False


if USE_GPU:
    import torch


def to_torch_array(x):
    return torch.tensor(x, dtype=torch.complex128, device="cuda")


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
        Specify the crossings in order, each entry is a 2-tuple (i,s), the elements have meanings:
            i : a strand index, the crossing is added between i and i+1
            s : the sign of the crossing, +1 or -1

    Returns
    -------
    float
    """
    writhe = sum([val[1] for val in crossing_list])
    n_qubits = num_strands + 1
    phi = (1 + np.sqrt(5)) / 2.0
    return (-np.exp(-1j*3*np.pi/5))**(3*writhe) * phi**(n_qubits - 2)


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
    U[2, 2] = np.exp(-1j * 4*np.pi / 5)
    U[3, 3] = np.exp(1j * 3*np.pi / 5)
    U[5, 5] = np.exp(1j * 4*np.pi / 5) / phi
    U[6, 6] = np.exp(1j * 3*np.pi / 5)
    U[7, 7] = -1 / phi
    U[7, 5] = np.exp(-1j * 3*np.pi / 5) / np.sqrt(phi)
    U[5, 7] = np.exp(-1j * 3*np.pi / 5) / np.sqrt(phi)

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
):
    """
    Make the MPO of a braid, does not include the projectors

    Parameters
    ----------
    num_strands : int
    crossing_list : iterable[tuple]
        Specify the crossings in order, each entry is a 2-tuple (i,s), the elements have meanings:
            i : a strand index, the crossing is added between i and i+1
            s : the sign of the crossing, +1 or -1
    max_bond : (optional) int
        Passed to the TN contraction when the MPO corresponding to each
        crossing are contracted together
    prog_bar : (optional) bool
        If set to True will display progress bar of building MPO
    debug_draw : (optional) bool
        If set to True will display drawings helpful for debugging

    Returns
    -------
    quimb.tensor.MatrixProductOperator
    """
    num_qubits = num_strands + 1

    U = make_braid_generator()
    Udag = U.T.conj()
    if USE_GPU:
        U = to_torch_array(U)
        Udag = to_torch_array(Udag)

    next_crossing = crossing_list[0]
    if next_crossing[1] == 1:
        tmp = U
    elif next_crossing[1] == -1:
        tmp = Udag
    else:
        raise ValueError(f'Cannot interpret sign of crossing: {next_crossing[1]}. Should have value +1 or -1.')
    
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

    for idx in _iterator:
        next_crossing = crossing_list[idx]
        if next_crossing[1] == 1:
            tmp = U
        elif next_crossing[1] == -1:
            tmp = Udag
        else:
            raise ValueError(f'Cannot interpret sign of crossing: {next_crossing[1]}. Should have value +1 or -1.')

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

    return braid_mpo


def add_projectors_to_braid_mpo(
    num_qubits,
    braid_mpo,
    max_bond=None,
    prog_bar=False,
    debug_draw=False,
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

    Returns
    -------
    quimb.tensor.MatrixProductOperator
    """
    proj_boundary_0, proj_bulk, proj_boundary_1 = make_projectors()
    if USE_GPU:
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

    if prog_bar:
        _iterator = tqdm(range(1, num_qubits-2))
    else:
        _iterator = range(1, num_qubits-2)

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

    return braid_mpo


def make_markov_closure_mpo(
    num_strands,
    crossing_list,
    max_bond=None,
    prog_bar=False,
    debug_draw=False,
):
    """
    Make the MPO of a braid including the projectors

    Parameters
    ----------
    num_strands : int
    crossing_list : iterable[tuple]
        Specify the crossings in order, each entry is a 2-tuple (i,s), the elements have meanings:
            i : a strand index, the crossing is added between i and i+1
            s : the sign of the crossing, +1 or -1
    max_bond : (optional) int
        Passed to the TN contraction when the MPO corresponding to each
        crossing are contracted together
    prog_bar : (optional) bool
        If set to True will display progress bar of building MPO
    debug_draw : (optional) bool
        If set to True will display drawings helpful for debugging

    Returns
    -------
    quimb.tensor.MatrixProductOperator
    """
    braid_mpo = make_braid_mpo(
        num_strands, crossing_list, max_bond=max_bond,
        prog_bar=prog_bar, debug_draw=debug_draw
    )
    return add_projectors_to_braid_mpo(
        num_strands + 1, braid_mpo, max_bond=max_bond, debug_draw=debug_draw
    )


def compute_weighted_trace_from_mpo(
    num_strands,
    crossing_list,
    max_bond=None,
    prog_bar=False,
    debug_draw=False,
):
    """
    Make and trace the MPO of a braid including the projectors

    Parameters
    ----------
    num_strands : int
    crossing_list : iterable[tuple]
        Specify the crossings in order, each entry is a 2-tuple (i,s), the elements have meanings:
            i : a strand index, the crossing is added between i and i+1
            s : the sign of the crossing, +1 or -1
    max_bond : (optional) int
        Passed to the TN contraction when the MPO corresponding to each
        crossing are contracted together
    prog_bar : (optional) bool
        If set to True will display progress bar of building MPO
    debug_draw : (optional) bool
        If set to True will display drawings helpful for debugging

    Returns
    -------
    numpy.complex128 (if using CPU backend)
        OR numpy.complex64 (if using GPU backend)
    """
    braid_mpo_with_projectors = make_markov_closure_mpo(
        num_strands, crossing_list, max_bond=max_bond,
        prog_bar=prog_bar, debug_draw=debug_draw
    )

    _trace = braid_mpo_with_projectors.trace()
    if USE_GPU:
        _trace = _trace.cpu().numpy()

    phi = (1 + np.sqrt(5)) / 2.0
    return _trace / phi ** (num_strands)


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
        raise ValueError(f'Cannot interpret sign of crossing: {next_crossing[1]}. Should have value +1 or -1.')

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
            raise ValueError(f'Cannot interpret sign of crossing: {next_crossing[1]}. Should have value +1 or -1.')

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


def _get_dense_from_mpo(mpo):
    """ """
    if USE_GPU:
        return mpo.to_dense().cpu().numpy()
    else:
        return mpo.to_dense()


def _run_tests():
    """ """

    if USE_GPU:
        rtol = 1e-4
        atol = 5e-6
    else:
        rtol = 1e-5
        atol = 1e-8
    
    #
    # test of make_braid_unitary_dense against matrices it should match in
    # small simple cases
    #

    U = make_braid_generator()
    Udag = U.T.conj()
    iden = np.array([
        [1, 0],
        [0, 1],
    ])

    # 2 strands
    assert np.allclose(make_braid_unitary_dense(2, [(0, +1)]), U)
    assert np.allclose(make_braid_unitary_dense(2, [(0, -1)]), Udag)
    assert np.allclose(make_braid_unitary_dense(2, [(0, +1), (0, +1)]), U @ U)
    assert np.allclose(make_braid_unitary_dense(2, [(0, +1), (0, -1)]), U @ Udag)
    assert np.allclose(make_braid_unitary_dense(2, [(0, -1), (0, -1)]), Udag @ Udag)
    assert np.allclose(
        make_braid_unitary_dense(2, [(0, +1), (0, +1), (0, +1)]), U @ U @ U)
    assert np.allclose(
        make_braid_unitary_dense(2, [(0, +1), (0, -1), (0, +1)]), U @ Udag @ U)
    assert np.allclose(
        make_braid_unitary_dense(2, [(0, -1), (0, +1), (0, -1)]), Udag @ U @ Udag)

    assert np.allclose(make_braid_unitary_dense(3, [(0, +1)]), np.kron(U, iden))
    assert np.allclose(
        make_braid_unitary_dense(3, [(0, +1), (0, +1)]), np.kron(U @ U, iden)
    )
    assert np.allclose(
        make_braid_unitary_dense(3, [(0, +1), (0, +1), (0, +1)]),
        np.kron(U @ U @ U, iden)
    )

    # 3 strands
    assert np.allclose(make_braid_unitary_dense(3, [(1, +1)]), np.kron(iden, U))
    assert np.allclose(
        make_braid_unitary_dense(3, [(1, +1), (1, +1)]), np.kron(iden, U @ U)
    )
    assert np.allclose(
        make_braid_unitary_dense(3, [(1, +1), (1, +1), (1, +1)]),
        np.kron(iden, U @ U @ U)
    )

    # 4 strands
    assert np.allclose(
        make_braid_unitary_dense(4, [(0, +1)]), np.kron(np.kron(U, iden), iden)
    )
    assert np.allclose(
        make_braid_unitary_dense(4, [(1, +1)]), np.kron(np.kron(iden, U), iden)
    )
    assert np.allclose(
        make_braid_unitary_dense(4, [(2, +1)]), np.kron(np.kron(iden, iden), U)
    )

    assert np.allclose(
        make_braid_unitary_dense(3, [(0, +1), (1, +1)]),
        np.kron(iden, U) @ np.kron(U, iden),
    )
    assert np.allclose(
        make_braid_unitary_dense(3, [(0, +1), (1, +1), (0, +1)]),
        np.kron(U, iden) @ np.kron(iden, U) @ np.kron(U, iden),
    )

    #
    # test of make_braid_mpo against make_braid_unitary_dense, by getting
    # dense repr from MPO
    #

    num_strands = 2
    crossing_list = [(0, +1)]
    assert np.allclose(
        _get_dense_from_mpo(make_braid_mpo(num_strands, crossing_list)),
        make_braid_unitary_dense(num_strands, crossing_list),
        rtol=rtol, atol=atol,
    )
    crossing_list = [(0, -1)]
    assert np.allclose(
        _get_dense_from_mpo(make_braid_mpo(num_strands, crossing_list)),
        make_braid_unitary_dense(num_strands, crossing_list),
        rtol=rtol, atol=atol,
    )
    crossing_list = [(0, +1), (0, +1)]
    assert np.allclose(
        _get_dense_from_mpo(make_braid_mpo(num_strands, crossing_list)),
        make_braid_unitary_dense(num_strands, crossing_list),
        rtol=rtol, atol=atol,
    )
    crossing_list = [(0, -1), (0, -1)]
    assert np.allclose(
        _get_dense_from_mpo(make_braid_mpo(num_strands, crossing_list)),
        make_braid_unitary_dense(num_strands, crossing_list),
        rtol=rtol, atol=atol,
    )

    num_strands = 3
    crossing_list = [(0, +1), (1, +1)]
    assert np.allclose(
        _get_dense_from_mpo(make_braid_mpo(num_strands, crossing_list)),
        make_braid_unitary_dense(num_strands, crossing_list),
        rtol=rtol, atol=atol,
    )
    crossing_list = [(0, +1), (1, +1), (0, +1)]
    assert np.allclose(
        _get_dense_from_mpo(make_braid_mpo(num_strands, crossing_list)),
        make_braid_unitary_dense(num_strands, crossing_list),
        rtol=rtol, atol=atol,
    )
    crossing_list = [(0, +1), (1, -1), (0, +1)]
    assert np.allclose(
        _get_dense_from_mpo(make_braid_mpo(num_strands, crossing_list)),
        make_braid_unitary_dense(num_strands, crossing_list),
        rtol=rtol, atol=atol,
    )
    crossing_list = [(0, +1), (1, -1), (0, -1)]
    assert np.allclose(
        _get_dense_from_mpo(make_braid_mpo(num_strands, crossing_list)),
        make_braid_unitary_dense(num_strands, crossing_list),
        rtol=rtol, atol=atol,
    )

    num_strands = 4
    crossing_list = [(0, +1)]
    assert np.allclose(
        _get_dense_from_mpo(make_braid_mpo(num_strands, crossing_list)),
        make_braid_unitary_dense(num_strands, crossing_list),
        rtol=rtol, atol=atol,
    )
    crossing_list = [(1, +1)]
    assert np.allclose(
        _get_dense_from_mpo(make_braid_mpo(num_strands, crossing_list)),
        make_braid_unitary_dense(num_strands, crossing_list),
        rtol=rtol, atol=atol,
    )
    crossing_list = [(2, +1)]
    assert np.allclose(
        _get_dense_from_mpo(make_braid_mpo(num_strands, crossing_list)),
        make_braid_unitary_dense(num_strands, crossing_list),
        rtol=rtol, atol=atol,
    )
    crossing_list = [(0, -1)]
    assert np.allclose(
        _get_dense_from_mpo(make_braid_mpo(num_strands, crossing_list)),
        make_braid_unitary_dense(num_strands, crossing_list),
        rtol=rtol, atol=atol,
    )
    crossing_list = [(1, -1)]
    assert np.allclose(
        _get_dense_from_mpo(make_braid_mpo(num_strands, crossing_list)),
        make_braid_unitary_dense(num_strands, crossing_list),
        rtol=rtol, atol=atol,
    )
    crossing_list = [(2, -1)]
    assert np.allclose(
        _get_dense_from_mpo(make_braid_mpo(num_strands, crossing_list)),
        make_braid_unitary_dense(num_strands, crossing_list),
        rtol=rtol, atol=atol,
    )

    #
    # test of compute_weighted_trace_from_dense against
    # known answer for small cases
    #

    for n in range(1, 15):
        assert len(gen_fibo_strings(n)) == fibo_number(n)

    phi = (1 + np.sqrt(5)) / 2.0

    num_strands = 2
    crossing_list = [(0, +1)]
    braid_dense = make_braid_unitary_dense(num_strands, crossing_list)
    expected_answer = braid_dense[2, 2] / phi**2 + braid_dense[3, 3] / phi
    assert np.isclose(
        compute_weighted_trace_from_dense(num_strands, crossing_list),
        expected_answer
    )
    crossing_list = [(0, -1)]
    braid_dense = make_braid_unitary_dense(num_strands, crossing_list)
    expected_answer = braid_dense[2, 2] / phi**2 + braid_dense[3, 3] / phi
    assert np.isclose(
        compute_weighted_trace_from_dense(num_strands, crossing_list),
        expected_answer
    )
    crossing_list = [(0, +1), (0, +1)]
    braid_dense = make_braid_unitary_dense(num_strands, crossing_list)
    expected_answer = braid_dense[2, 2] / phi**2 + braid_dense[3, 3] / phi
    assert np.isclose(
        compute_weighted_trace_from_dense(num_strands, crossing_list),
        expected_answer
    )
    crossing_list = [(0, -1), (0, +1)]
    braid_dense = make_braid_unitary_dense(num_strands, crossing_list)
    expected_answer = braid_dense[2, 2] / phi**2 + braid_dense[3, 3] / phi
    assert np.isclose(
        compute_weighted_trace_from_dense(num_strands, crossing_list),
        expected_answer
    )
    crossing_list = [(0, +1), (0, +1), (0, +1)]
    braid_dense = make_braid_unitary_dense(num_strands, crossing_list)
    expected_answer = braid_dense[2, 2] / phi**2 + braid_dense[3, 3] / phi
    assert np.isclose(
        compute_weighted_trace_from_dense(num_strands, crossing_list),
        expected_answer
    )

    num_strands = 3
    crossing_list = [(0, +1), (1, +1)]
    braid_dense = make_braid_unitary_dense(num_strands, crossing_list)
    expected_answer = (
        braid_dense[5, 5] / phi**2
        + braid_dense[6, 6] / phi**3
        + braid_dense[7, 7] / phi**2
    )
    assert np.isclose(
        compute_weighted_trace_from_dense(num_strands, crossing_list),
        expected_answer
    )
    crossing_list = [(0, +1), (1, -1)]
    braid_dense = make_braid_unitary_dense(num_strands, crossing_list)
    expected_answer = (
        braid_dense[5, 5] / phi**2
        + braid_dense[6, 6] / phi**3
        + braid_dense[7, 7] / phi**2
    )
    assert np.isclose(
        compute_weighted_trace_from_dense(num_strands, crossing_list),
        expected_answer
    )
    crossing_list = [(0, +1), (1, +1), (0, +1)]
    braid_dense = make_braid_unitary_dense(num_strands, crossing_list)
    expected_answer = (
        braid_dense[5, 5] / phi**2
        + braid_dense[6, 6] / phi**3
        + braid_dense[7, 7] / phi**2
    )
    assert np.isclose(
        compute_weighted_trace_from_dense(num_strands, crossing_list),
        expected_answer
    )
    crossing_list = [(0, +1), (1, -1), (0, +1)]
    braid_dense = make_braid_unitary_dense(num_strands, crossing_list)
    expected_answer = (
        braid_dense[5, 5] / phi**2
        + braid_dense[6, 6] / phi**3
        + braid_dense[7, 7] / phi**2
    )
    assert np.isclose(
        compute_weighted_trace_from_dense(num_strands, crossing_list),
        expected_answer
    )
    crossing_list = [(0, +1), (1, -1), (0, -1)]
    braid_dense = make_braid_unitary_dense(num_strands, crossing_list)
    expected_answer = (
        braid_dense[5, 5] / phi**2
        + braid_dense[6, 6] / phi**3
        + braid_dense[7, 7] / phi**2
    )
    assert np.isclose(
        compute_weighted_trace_from_dense(num_strands, crossing_list),
        expected_answer
    )

    #
    # test of compute_weighted_trace_from_mpo against
    # compute_weighted_trace_from_dense
    #

    num_strands = 3
    crossing_list = [(0, +1), (1, +1)]
    assert np.isclose(
        compute_weighted_trace_from_mpo(num_strands, crossing_list),
        compute_weighted_trace_from_dense(num_strands, crossing_list),
    )
    crossing_list = [(0, +1), (1, +1), (0, +1)]
    assert np.isclose(
        compute_weighted_trace_from_mpo(num_strands, crossing_list),
        compute_weighted_trace_from_dense(num_strands, crossing_list),
    )
    crossing_list = [(0, +1), (1, -1), (0, +1)]
    assert np.isclose(
        compute_weighted_trace_from_mpo(num_strands, crossing_list),
        compute_weighted_trace_from_dense(num_strands, crossing_list),
    )
    crossing_list = [(0, +1), (1, -1), (0, -1)]
    assert np.isclose(
        compute_weighted_trace_from_mpo(num_strands, crossing_list),
        compute_weighted_trace_from_dense(num_strands, crossing_list),
    )

    num_strands = 4
    crossing_list = [(0, +1), (1, +1), (2, +1)]
    assert np.isclose(
        compute_weighted_trace_from_mpo(num_strands, crossing_list),
        compute_weighted_trace_from_dense(num_strands, crossing_list),
    )
    crossing_list = [(0, -1), (1, +1), (2, +1)]
    assert np.isclose(
        compute_weighted_trace_from_mpo(num_strands, crossing_list),
        compute_weighted_trace_from_dense(num_strands, crossing_list),
    )
    crossing_list = [(0, +1), (1, -1), (2, -1)]
    assert np.isclose(
        compute_weighted_trace_from_mpo(num_strands, crossing_list),
        compute_weighted_trace_from_dense(num_strands, crossing_list),
    )
    crossing_list = [(0, +1), (1, +1), (2, +1), (0, +1)]
    assert np.isclose(
        compute_weighted_trace_from_mpo(num_strands, crossing_list),
        compute_weighted_trace_from_dense(num_strands, crossing_list),
    )

    # 
    # 7, 4 knot
    # Compare the value we calculate to the value from knot-atlas (the value on knot-atlas is wrong, we need to take the complex conjugate)
    # 

    def jones_7_4(q):
        return -q**8 + q**7 - 2*q**6 + 3*q**5 - 2*q**4 + 3*q**3 - 2*q**2 + q
    
    crossing_list = [
        (0, +1),
        (0, +1),
        (1, +1),
        (0, -1),
        (1, +1),
        (1, +1),
        (2, +1),
        (1, -1),
        (2, +1),
    ]
    
    # writhe = sum([val[1] for val in crossing_list])
    # n_qubits = 5
    # phi = (1 + np.sqrt(5)) / 2.0
    # factors = (-np.exp(-1j*3*np.pi/5))**(3*writhe) * phi**(n_qubits - 2)

    n_qubits = 5
    factors = jones_factor(n_qubits-1, crossing_list)
    
    mpo_est = factors * compute_weighted_trace_from_mpo(n_qubits-1, crossing_list)

    # print(jones_7_4(np.exp(-1j*2*np.pi/5)), mpo_est)
    # print(jones_7_4(np.exp(-1j*2*np.pi/5)) - mpo_est, rtol, atol,)
    assert np.isclose(
        jones_7_4(np.exp(-1j*2*np.pi/5)) - mpo_est,
        0, rtol=rtol, atol=atol,
    )


if __name__ == "__main__":

    if RUN_TESTS:
        print('\nRunning tests...')
        _run_tests()
        print('DONE.\n')

    # # max bond dimension for MPO
    # max_bond = 2**12

    # # number of fibonacci wires
    # num_strands = 4

    # # knot crossings
    # crossing_list = [
    #     (0, +1),
    #     (2, +1),
    #     (1, -1),
    #     (0, +1),
    #     (0, +1),
    #     (2, +1),
    #     (1, -1),
    # ]

    # # compute the weighted trace with fixed max bond dim for braid in Fig11
    # mpo_est = compute_weighted_trace_from_mpo(
    #     num_strands, crossing_list, prog_bar=True, max_bond=max_bond,)
    # print(f'weighted trace of knot in Fig11: {mpo_est}')
