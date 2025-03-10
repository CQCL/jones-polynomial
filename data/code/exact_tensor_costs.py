import numpy as np
import cotengra
import quimb.tensor as qtn
import pickle
import sys
import os

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

def make_braid_tn(braid, sv=False):
    U = make_braid_generator()
    proj_boundary_0, proj_bulk, proj_boundary_1 = make_projectors()
    circ = qtn.Circuit(braid['strands'] + 1)
    for (idx, sign) in braid['crossing_list']:
        circ.apply_gate(U if sign > 0 else U.T.conj(), idx, idx+1, idx+2, gate_round=0)
    if not sv:
        for idx in range(braid['strands']):
            if idx == 0:
                circ.apply_gate(proj_boundary_0, idx, idx+1, gate_round=1)
            elif idx == braid['strands'] - 1:
                circ.apply_gate(proj_boundary_1, idx, idx+1, gate_round=1)
            else:
                circ.apply_gate(proj_bulk, idx, idx+1, gate_round=1)
        network: qtn.TensorNetwork = circ.get_uni()
        network.reindex({f'b{i}': f'k{i}' for i in range(braid['strands'] + 1)}, inplace=True)
    else:
        network: qtn.TensorNetwork = circ.amplitude(b='0'*(braid['strands'] + 1), rehearse='tn', simplify_sequence='')
    return network

processes = int(sys.argv[1])
procid = int(sys.argv[2])

def process_braid(val):
    idx, braid = val
    print(len(braid['crossing_list']), braid['strands'], np.log10(float(16 * len(braid['crossing_list']) * 2 ** (2*(braid['strands'] + 1)))))
    network = make_braid_tn(braid, sv=False)
    hyper = cotengra.HyperOptimizer(methods=['kahypar', 'greedy'], minimize='flops', optlib='optuna', parallel=False, max_repeats=128, on_trial_error='raise', progbar=False)
    info = network.contraction_info(hyper)
    nbraid = braid.copy()
    nbraid['path_info'] = { k: getattr(info, k) for k in ['largest_intermediate', 'opt_cost', 'path'] }
    return idx, nbraid

if __name__ == "__main__":
    braids = pickle.load(open("../raw_data/braids_n10_29_l5_50_c50.pkl", "rb"))
    os.mkdir('exact_tensor_costs_outputs')
    for val in list(enumerate(braids))[procid::processes][::-1]:
        idx, nbraid = process_braid(val)
        pickle.dump(nbraid, open(f"exact_tensor_costs_outputs/b{idx:04}.pkl", "wb"))
