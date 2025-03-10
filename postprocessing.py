import argparse
import rich_argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=rich_argparse.RichHelpFormatter)
    parser.add_argument("BITS", help="file containing sampled classical input bits")
    parser.add_argument("DATA", help="data file containing measurement outcomes")
    parser.add_argument("BRAID", help="braid json file to consider")
    parser.add_argument("-o", "--out-dir", help="directory to put output files", default=".")
    parser.add_argument("-n", "--no-mitigation", help="disable error mitigation (increases bias but may decrease variance)", action="store_true")
    parser.add_argument("-b", "--bootstrap", help="compute confidence intervals using bootstrapping", action="store_true")
    parser.add_argument("-r", "--resamples", help="number of resamples to use for bootstrapping", type=int, default=10000)
    parser.add_argument("-c", "--confidence", help="confidence level to calculate intervals for", type=float, default=0.05)
    args = parser.parse_args()

import numpy as np
import json
import common
import scipy.stats
import pathlib
import tqdm
import multiprocessing

phi = (1 + 5**0.5)/2

def bootstrap_ci(replicas: np.ndarray, alpha: float):
    n = replicas.shape[0]
    if n == 0:
        return (0.0, 0.0)
    elif n == 1:
        return (replicas[0], replicas[0], 0.0)
    alpha2 = scipy.stats.norm.cdf(np.sqrt(n / (n - 1)) * scipy.stats.t.ppf(alpha / 2, n - 1))
    return tuple(np.quantile(replicas, [alpha2, 1.0 - alpha2])) + (np.std(replicas),)

def do_replica(arg):
    i, bits_orig, data_orig, braid, mitigation = arg
    if i == 0:
        data, bits = data_orig, bits_orig
    else:
        indices = np.random.randint(bits_orig.shape[0], size=bits_orig.shape[0])
        data, bits = data_orig[indices, :], bits_orig[indices, :]
        
    mask = (np.all(data[:, 2:] == 0, axis=1) & (data[:, 0] == 0)).astype(float)
    sign = 1 - 2 * data[:, 1].astype(float)
    value = mask * sign

    if mitigation:
        xbits = data[:, 2:] ^ bits[:, 1:]
        reject = (data[:, 0] != 0) | np.any((xbits[:, :-1] | xbits[:, 1:]) == 0, axis=1)
        reals = value[(bits[:, 0] == 0) & ~reject]
        imags = value[(bits[:, 0] == 1) & ~reject]
        dr = np.mean(reject.astype(float), axis=0)
    else:
        reals = value[bits[:, 0] == 0]
        imags = value[bits[:, 0] == 1]
        dr = 0.0

    est_raw = np.mean(reals, axis=0) + 1j * np.mean(imags, axis=0)
    est = (-np.exp(-1j*3*np.pi/5))**(3*braid.writhe) * phi ** (data.shape[-1] - 2) * est_raw
    
    return i, est, est_raw, dr

def compute_jones(bits: np.ndarray, data: np.ndarray, braid: common.Braid, mitigation: bool, bootstrap: bool, resamples: int = 100, alpha: float = 0.05):
    if data.shape[1] != 1:
        print("Warning: expected DATA to be an array of shape n x 1 x m, using element 0 along axis 1 (multi-shot simulations are not supported)")
    data = data[:, 0, :]
    data_orig = data
    bits_orig = bits

    if bootstrap:
        pbar = tqdm.trange(resamples + 1, desc="processing replicas")
    else:
        pbar = range(1)
        resamples = 0

    ests = [None]*(resamples + 1)
    ests_raw = [None]*(resamples + 1)
    drs = [None]*(resamples + 1)
    pool = multiprocessing.Pool(multiprocessing.cpu_count() // 2)
    for i, est, est_raw, dr in pool.imap_unordered(do_replica, ((i, bits_orig, data_orig, braid, mitigation) for i in pbar)):
        ests[i] = est
        ests_raw[i] = est_raw
        drs[i] = dr
    est = np.array(ests)
    est_raw = np.array(ests_raw)
    dr = np.array(drs)

    return est[0], bootstrap_ci(est.real, alpha), bootstrap_ci(est.imag, alpha), dr[0], bootstrap_ci(dr, alpha), est_raw[0], bootstrap_ci(est_raw.real, alpha), bootstrap_ci(est_raw.imag, alpha)

if __name__ == "__main__":
    with open(args.BRAID, "r") as f:
        braid = json.load(f)
    braid = common.Braid.from_word(braid["optimized"]["word"])
    data = np.load(args.DATA)
    bits = np.load(args.BITS)
    ofile = pathlib.Path(args.out_dir) / f'{pathlib.Path(args.DATA).stem}.res.json'

    est, est_re_ci, est_im_ci, dr, dr_ci, est_raw, est_raw_re_ci, est_raw_im_ci = compute_jones(
        bits, data, braid, 
        not args.no_mitigation, args.bootstrap, 
        args.resamples, args.confidence
    )

    with open(ofile, "w") as f:
        json.dump({
            "jones_real": {
                "value": est.real,
                "ci_low": est_re_ci[0],
                "ci_high": est_re_ci[1],
                "std_dev": est_re_ci[2]
            },
            "jones_imag": {
                "value": est.imag,
                "ci_low": est_im_ci[0],
                "ci_high": est_im_ci[1],
                "std_dev": est_im_ci[2]
            },
            "raw_real": {
                "value": est_raw.real,
                "ci_low": est_raw_re_ci[0],
                "ci_high": est_raw_re_ci[1],
                "std_dev": est_raw_re_ci[2]
            },
            "raw_imag": {
                "value": est_raw.imag,
                "ci_low": est_raw_im_ci[0],
                "ci_high": est_raw_im_ci[1],
                "std_dev": est_raw_im_ci[2]
            },
            "discard_rate": {
                "value": dr,
                "ci_low": dr_ci[0],
                "ci_high": dr_ci[1],
                "std_dev": dr_ci[2]
            },
            "confidence_level": args.confidence,
            "resamples": args.resamples if args.bootstrap else 0,
            "mitigation": not args.no_mitigation,
            "braid": args.BRAID,
            "bits": args.BITS,
            "data": args.DATA
        }, f, indent=4)
        print(f"wrote `{ofile}`")



