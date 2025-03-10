import argparse
import rich_argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=rich_argparse.RichHelpFormatter)
    parser.add_argument("SHOTS", help="number of shots per braid", type=int, default=1000)
    parser.add_argument("BRAID", help="braid json files to sample for", nargs="+")
    parser.add_argument("-o", "--out-dir", help="directory to put output files", default=".")
    args = parser.parse_args()

from common import Braid, phi, fib
import json
import pathlib
import random
import numpy as np

def sample_bits(n: int, shots: int) -> list[list[bool]]:
    output = []
    for _ in range(shots // 2):
        bits = [False]*n
        bits[1] = True
        bits[-1] = True if random.random() < (fib(n - 1) / phi ** (n - 2)) else False
        if not bits[-1]:
            bits[-2] = True
            k = random.randrange(fib(n - 2))
            start = n - 3
        else:
            k = random.randrange(fib(n - 1))
            start = n - 2
        
        for i in range(start, 1, -1):
            if k >= fib(i):
                bits[i] = False
                k -= fib(i)
            else:
                bits[i] = True

        output.append([0] + bits[2:])
        output.append([1] + bits[2:])
    return output

if __name__ == "__main__":
    for fname in args.BRAID:
        with open(fname, "r") as f:
            bspec = json.load(f)
        braid = Braid.from_word(bspec["optimized"]["word"])
        bits = np.array(sample_bits(braid.strands + 1, args.SHOTS), dtype=bool)
        ofile = pathlib.Path(args.out_dir) / f'{pathlib.Path(fname).stem}.npy'
        np.save(ofile, bits)
        print(f"wrote `{ofile}`")
