import tqdm
import numpy as np
import multiprocessing
import pickle

def get_value_single(bits, data, mitigation=False):
    mask = (np.all(data[2:] == 0) & (data[0] == 0)).astype(float)
    sign = 1 - 2 * data[1].astype(float)
    value = mask * sign
    if mitigation:
        xbits = data[2:] ^ bits[1:]
        reject = (data[0] != 0) | np.any((xbits[:-1] | xbits[1:]) == 0)
    else:
        reject = False
    return value, reject

def undo_depolarizing(re, im, d):
    rec = ((2 - d)*re + d*im) / (2*(d - 1)**2) 
    imc = ((2 - d)*im + d*re) / (2*(d - 1)**2)
    return np.sqrt((rec - 2)/4+1e-10) + 1j*np.sqrt((imc - 2)/4+1e-10)

def minimal_d(x, y):
    return max(1 + abs(x - y)/8 - np.sqrt((x - y)**2 + 16*(x + y))/8, 0)

def do_ests(bits, data_br, data_bi, data_cr, data_ci):
    mitigation = False
    res = []
    ims = []
    bs = []
    cs = []
    for i in range(bits.shape[0]):
        v_br, r1 = get_value_single(bits[i], data_br[i], mitigation)
        v_bi, r2 = get_value_single(bits[i], data_bi[i], mitigation)
        v_cr, r3 = get_value_single(bits[i], data_cr[i], mitigation)
        v_ci, r4 = get_value_single(bits[i], data_ci[i], mitigation)
        v_b = v_br + 1j*v_bi
        v_c = v_cr + 1j*v_ci
        if mitigation and (r1 | r2 | r3 | r4):
            continue
        res.append((v_br + v_cr)**2 + (v_bi + v_ci)**2)
        ims.append((v_br - v_cr)**2 + (v_bi - v_ci)**2)
        bs.append(v_b)
        cs.append(v_c)
    re, im = np.mean(res), np.mean(ims)

    mitigation = True
    res2 = []
    ims2 = []
    bs2 = []
    cs2 = []
    for i in range(bits.shape[0]):
        v_br, r1 = get_value_single(bits[i], data_br[i], mitigation)
        v_bi, r2 = get_value_single(bits[i], data_bi[i], mitigation)
        v_cr, r3 = get_value_single(bits[i], data_cr[i], mitigation)
        v_ci, r4 = get_value_single(bits[i], data_ci[i], mitigation)
        v_b = v_br + 1j*v_bi
        v_c = v_cr + 1j*v_ci
        if mitigation and (r1 | r2 | r3 | r4):
            continue
        res2.append((v_br + v_cr)**2 + (v_bi + v_ci)**2)
        ims2.append((v_br - v_cr)**2 + (v_bi - v_ci)**2)
        bs2.append(v_b)
        cs2.append(v_c)
    re2, im2 = np.mean(res2), np.mean(ims2)

    avg_then_mit_nf = -abs(np.mean(bs) + np.mean(cs))/2 - 1j*abs(np.mean(bs) - np.mean(cs))/2
    avg_then_mit = -abs(np.mean(bs2) + np.mean(cs2))/2 - 1j*abs(np.mean(bs2) - np.mean(cs2))/2
    mit_then_avg = -undo_depolarizing(re2, im2, minimal_d(re2, im2))
    mit_then_avg_nf = -undo_depolarizing(re, im, minimal_d(re, im))

    b_nomit = np.mean(bs)
    b_mit = np.mean(bs2)
    c_nomit = np.mean(cs)
    c_mit = np.mean(cs2)

    return b_nomit, c_nomit, b_mit, c_mit, avg_then_mit_nf, avg_then_mit, mit_then_avg_nf, mit_then_avg

def do_rep(arg):
    bits, data_br, data_bi, data_cr, data_ci = arg
    indices = np.random.randint(bits.shape[0], size=bits.shape[0])
    rquants = do_ests(bits[indices], data_br[indices], data_bi[indices], data_cr[indices], data_ci[indices])
    return rquants

def do_bootstrap(bits, data_br, data_bi, data_cr, data_ci, nreps, nprocs):
    quants = do_ests(bits, data_br, data_bi, data_cr, data_ci)
    pool = multiprocessing.Pool(nprocs)
    reps = list(tqdm.tqdm(pool.imap(do_rep, ((bits, data_br, data_bi, data_cr, data_ci) for _ in range(nreps))), total=nreps))
    reps = np.stack(reps, axis=1)
    return list(zip(quants, reps))

if __name__ == "__main__":
    import json
    import glob
    shots = []
    for f in glob.glob(f'results/h2-2/*.json'):
        with open(f, 'r') as file:
            dict_results = json.load(file)
            from pytket.backends.backendresult import BackendResult
            res = BackendResult.from_dict(dict_results)
            shots.append(res.get_shots())

    bits = np.concatenate([s[:, :15] for s in shots], axis=0).astype(bool)
    entropy = np.concatenate([s[:, 15:31] for s in shots], axis=0).astype(bool)
    data_bi = np.concatenate([s[:, 31:47] for s in shots], axis=0).astype(bool)
    data_br = np.concatenate([s[:, 47:63] for s in shots], axis=0).astype(bool)
    data_ci = np.concatenate([s[:, 63:79] for s in shots], axis=0).astype(bool)
    data_cr = np.concatenate([s[:, 79:95] for s in shots], axis=0).astype(bool)

    res = do_bootstrap(bits, data_br, data_bi, data_cr, data_ci, 100000, 64)
    pickle.dump(res, open("../processed/h2-2_interleaved.pkl", "wb"))