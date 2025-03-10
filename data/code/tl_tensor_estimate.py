import pickle
import os.path
import os
import sys
import tl_tensor
import cotengra
import opt_einsum

processes = int(sys.argv[1])
procid = int(sys.argv[2])

def process_braid(val):
    idx, braid = val
    print(idx, len(braid['crossing_list']), braid['strands'])
    network = tl_tensor.TLTensorNetwork.from_word([sign * (pos + 1) for pos, sign in braid['crossing_list']], strands=braid['strands'], normalize=False)
    qnetwork = network._proxy_qtn()
    hyper = cotengra.HyperOptimizer(['kahypar', 'greedy'], minimize='flops', max_repeats=128, progbar=False, parallel=False)
    info = qnetwork.contraction_tree(optimize=hyper)
    nbraid = braid.copy()
    nbraid['path_info'] = { 'path': info.get_path(), 'cost': info.total_flops(log=10), 'max_intermediate': info.max_size(log=2) }
    print(idx, nbraid['path_info']['cost'], nbraid['path_info']['max_intermediate'])
    return idx, nbraid

if __name__ == "__main__":
    braids = pickle.load(open("../raw_data/braids_n10_29_l5_50_c50.pkl", "rb"))
    os.mkdir('tl_tensor_estimate_outputs')
    braids = [(idx, braid) for idx, braid in enumerate(braids) if not os.path.exists(f"exact_outputs_tl2/b{idx:04}.pkl",)]
    for val in braids[procid::processes][::-1]:
        idx, nbraid = process_braid(val)
        pickle.dump(nbraid, open(f"tl_tensor_estimate_outputs/b{idx:04}.pkl", "wb"))
