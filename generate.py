import argparse
import rich_argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=rich_argparse.RichHelpFormatter)
    parser.add_argument("-o", "--out-dir", help="directory to place output files", default=".")
    parser.add_argument("-m", "--minimize", help="algorithm to use during minimization", choices=['full', 'peephole', 'cancel', 'none'], default='full')
    parser.add_argument("-s", "--steps", help="number of random moves to consider during minimization", type=int, default=100)
    parser.add_argument("--non-cyclic", help="during minimization, do not assume the braid is closed", action="store_true")
    subparsers = parser.add_subparsers(dest='command', required=True)
    subparser = subparsers.add_parser('create', help='create a file for a specific braid or link', formatter_class=rich_argparse.RichHelpFormatter)
    group = subparser.add_mutually_exclusive_group(required=True)
    group.add_argument("-w", "--word", help="create a braid from a braid word")
    group.add_argument("-p", "--planar", help="create a braid from a planar diagram code")
    group.add_argument("-d", "--dt-code", help="create a braid from a Dowker-Thistlethwaite code")
    group.add_argument("-n", "--name", help="create a braid from a named link")
    subparser = subparsers.add_parser('random', help='generate random brickwall braids', formatter_class=rich_argparse.RichHelpFormatter)
    subparser.add_argument("strands", help="number of strands for braids", type=int)
    subparser.add_argument("layers", help="number of layers in brickwall", type=int)
    subparser.add_argument("-c", "--count", help="number of random braids to generate", type=int, default=1)
    subparser.add_argument("-p", "--prob", help="probability of each generator being present", type=float, default=0.666)
    subparser.add_argument("-u", "--uniform", help="uniform sampling of braid words rather than brickwall", action="store_true")
    subparser = subparsers.add_parser('benchmark', help='generate a set of benchmark brickwall braids', formatter_class=rich_argparse.RichHelpFormatter)
    subparser.add_argument("strands", help="number of strands for braids", type=int)
    subparser.add_argument("layers", help="number of layers in brickwall", type=int)
    subparser.add_argument("-c", "--count", help="number of random braids to generate", type=int, default=1)
    subparser.add_argument("-i", "--inner", help="add an inner layer of crossings to randomize the expectation value", action="store_true")
    subparser.add_argument("-p", "--prob", help="probability of each generator being present", type=float, default=0.666)
    args = parser.parse_args()

import random
import cmath
from common import Braid, Generator, phi
import minimize
import time
import tqdm
import json
import pathlib
import numpy as np
import scipy
import scipy.optimize as opt
import functools

def torus(p: int, q: int) -> tuple[Braid, complex]:
    braid = Braid.from_word([
        i if q > 0 else -i for i in range(1, p)
    ] * abs(q))
    t = cmath.exp(1j * cmath.pi * 2 / 5)
    jones = t ** (((p - 1)*(q - 1))/2) * (1 - t ** (p+1) - t ** (q + 1) + t ** (p + q)) / (1 - t*t)
    return braid, jones

def random3(l: int, t: int) -> tuple[Braid, complex]:
    braid = Braid.from_word(random.choices([t, -t, (t+1), -(t+1)], k=l))

    A = cmath.exp((3/5)*1j*np.pi)
    Ainv = cmath.exp((-3/5)*1j*np.pi)
    delta = (cmath.sqrt(5) + 1) / 2
    a = 1 / delta
    b = cmath.sqrt(1 - a ** 2)
    U = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, Ainv*delta + A, 0, 0, 0, 0, 0],
        [0, 0, 0, A, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, Ainv*a + A, 0, Ainv*b],
        [0, 0, 0, 0, 0, 0, A, 0],
        [0, 0, 0, 0, 0, Ainv*b, 0, Ainv*delta*b**2 + A]
    ], dtype=np.complex128)
    U_braid = np.eye(16, 16)
    for g in braid.gens:
        U_g = np.kron(np.kron(np.eye(2**(g.pos - t)), (U if g.sign > 0 else U.T.conj())), np.eye(2**((t + 1) - g.pos)))
        U_braid = U_g @ U_braid
    raw_val = ((U_braid[0b010, 0b010] + U_braid[0b110, 0b110]) + (U_braid[0b011, 0b011] + U_braid[0b101, 0b101] + U_braid[0b111, 0b111])*delta)/(2 + 3*delta)
    jones = (-np.exp(-1j*3*np.pi/5))**(3*braid.writhe) * delta ** 2 * raw_val

    return braid, jones


def augment(braid: Braid, layers: int, p: float = 0.666, strands: int = -1) -> Braid:
    if strands < 0:
        strands = braid.strands
    pre = []
    post = []
    for l in range(layers):
        for i in range(1 + l % 2, strands, 2):
            r = random.random()
            if r < p/2:
                pre.append(Generator(i))
                post.append(Generator(-i))
            elif r < p:
                pre.append(Generator(-i))
                post.append(Generator(i))
    post.reverse()
    nbraid = Braid(
        pre + braid.copy().gens + post
    )
    return nbraid

def brickwall(strands: int, layers: int, p=0.666) -> Braid:
    word = []
    for l in range(layers):
        if l % 2 == 0:
            start = 1
        else:
            start = 2
        for idx in range(start, strands, 2):
            r = random.random()
            if r < p/2:
                word.append(idx)
            elif r < p:
                word.append(-idx)
    return Braid.from_word(word)

def uniform(strands: int, gens: int) -> Braid:
    word = []
    for _ in range(gens):
        g = random.randrange(1, strands)
        g = -g if random.random() < 0.5 else g
        word.append(g)
    return Braid.from_word(word)

# Compute the expectation value of a 3-stranded braidword
def jones3_unweighted(braid: list[int]):
    braid = Braid.from_word(braid)
    A = cmath.exp((3/5)*1j*np.pi)
    Ainv = cmath.exp((-3/5)*1j*np.pi)
    delta = (cmath.sqrt(5) + 1) / 2
    a = 1 / delta
    b = cmath.sqrt(1 - a ** 2)
    U = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, Ainv*delta + A, 0, 0, 0, 0, 0],
        [0, 0, 0, A, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, Ainv*a + A, 0, Ainv*b],
        [0, 0, 0, 0, 0, 0, A, 0],
        [0, 0, 0, 0, 0, Ainv*b, 0, Ainv*delta*b**2 + A]
    ], dtype=np.complex128)
    U_braid = np.eye(16, 16)
    for g in braid.gens:
        U_g = np.kron(np.kron(np.eye(2**(g.pos - 1)), (U if g.sign > 0 else U.T.conj())), np.eye(2**(2 - g.pos)))
        U_braid = U_g @ U_braid
    
    raw_val = ((U_braid[0b0101, 0b0101] + U_braid[0b0111, 0b0111])*phi + U_braid[0b0110, 0b0110])

    return raw_val / phi**3


rs = [1.0, 0.9539, 0.8208, 0.618, 0.5895, 0.4721, 0.382, 0.2361, 0.1459]
bs = [
    [()],
    [(2, 2, 2), (1, 1, 1), (-2, -2, -2), (-1, -1, -1)],
    [(-1, -1, -1, -1), (1, 1, 1, 1), (2, 2, 2, 2), (-2, -2, -2, -2)],
    [(-2, -1, 2), (1,), (-1,), (-2,), (-1, -2, 1), (1, 2, -1), (2, 1, -2), (2,)],
    [(1, 2, 1, 1), (2, 1, 1, 1), (-1, -2, -1, -1), (-1, -2, -2, -2), (-1, -1, -1, -2),
     (-2, -1, -2, -2), (-2, -2, -2, -1), (-2, -1, -1, -1), (2, 2, -1, 2), (1, 1, 1, -2),
     (1, 1, -2, 1), (-2, -2, 1, -2), (2, 2, 2, 1), (-1, -1, -1, 2), (-1, 2, -1, -1), (-2, 1, 1, 1),
     (2, -1, 2, 2), (-1, 2, 2, 2), (2, -1, -1, -1), (1, 1, 1, 2), (-2, -2, -2, 1), (2, 2, 2, -1),
     (1, -2, -2, -2), (-1, -1, 2, -1), (1, 2, 2, 2), (2, 1, 2, 2), (-2, 1, -2, -2), (1, -2, 1, 1)],
    [(1, -2, 1, -2), (2, -1, 2, -1), (-1, 2, -1, 2), (-2, 1, -2, 1)],
    [(2, 2), (1, 2, -1, 2), (2, 2, 1, -2), (-1, -1), (-2, -1), (-1, -2), (-2, -2), (-1, -2, 1, -2),
     (-1, 2), (1, 1, 2, -1), (-2, -1, -1, 2), (-2, -2, -1, 2), (-2, -1, 2, 2), (2, 1, 1, -2), (1, -2),
     (-1, -2, -2, 1), (-1, -1, -2, 1), (1, 2), (2, 1), (2, 1, -2, 1), (1, 2, 2, -1), (-2, 1), 
     (-2, -1, 2, -1), (2, -1), (1, 2, -1, -1), (2, 1, -2, -2), (1, 1), (-1, -2, 1, 1)],
    [(1, -2, 1), (-1, 2, 2), (2, 2, -1), (1, 1, 2), (-1, -1, 2), (-2, 1, 1), (2, -1, -1), (1, 2, 1),
     (2, 1, 1), (2, -1, 2), (-2, 1, -2), (1, -2, -2), (-2, -2, 1), (-1, 2, -1), (1, 2, 2), (-2, -2, -1),
     (-1, -1, -2), (-1, -2, -1), (-1, -2, -2), (-2, -1, -1), (2, 2, 1), (1, 1, -2)],
    [(-2, -1, -1, -2), (-2, -2, -1, -1), (-1, -1, 2, 2), (-1, 2, 2, -1), (1, 1, -2, -2),
     (1, 2, 2, 1), (2, 1, 1, 2), (1, -2, -2, 1), (-2, 1, 1, -2), (2, -1, -1, 2), (1, 1, 2, 2),
     (2, 2, -1, -1), (2, 2, 1, 1), (-1, -1, -2, -2), (-1, -2, -2, -1), (-2, -2, 1, 1)]
]


# Find a distribution over 3-stranded braids such that the magnitude of the product
# of the expectation values of n braids is uniformly distributed in [0, 1]
@functools.lru_cache
def find_dist(n):
    # Solve for the weights of each braid by comparing the moment
    # generating function of the actual and desired distributions
    tvals = np.linspace(0, 7.5, 100)
    b = (1 / (tvals + 1)) ** (1/n)
    A = np.array(rs)[None, :] ** tvals[:, None]

    def target(x):
        return np.linalg.norm(A @ x - b)

    def target_jac(x):
        return 2 * (A.T @ (A @ x - b))

    def con(x):
        return np.sum(x) - 1
    
    def con_jac(x):
        return np.ones_like(x)
    
    bounds = [(0.0, None)]*len(rs)
    constraint = {
        'type': 'eq',
        'fun': con,
        'jac': con_jac
    }

    x0 = np.random.random(len(rs))
    x0 /= np.sum(x0)

    res = opt.minimize(target, x0, jac=target_jac, bounds=bounds, constraints=constraint, method='SLSQP', options={'maxiter': 1000}, tol=1e-10)
    return res.x

# Sample a single brickwall layer comprised of a tensor of
# 3-stranded braids where the magnitude of the expectation value
# of the closure is uniformly distributed between [0, 1]
def random3_layer(strands: int) -> tuple[list[Generator], complex]:
    n = strands // 3
    weights = find_dist(n)
    r_choices = random.choices(list(range(len(rs))), weights=weights, k=n)
    braids = [random.choice(bs[r]) for r in r_choices]
    # Ensure there is a non-zero braid at the bottom if possible
    # to try and make the overall layer as wide as possible
    if len(braids[-1]) == 0 and max(len(b) for b in braids) > 0:
        idx = random.choice([i for i, b in enumerate(braids) if len(b) > 0])
        braids[idx], braids[-1] = braids[-1], braids[idx]
    expval = np.prod([jones3_unweighted(b) for b in braids])
    offset_braid = []
    for i, b in enumerate(braids):
        for g in b:
            if g > 0:
                offset_braid.append(Generator(g + 3*i))
            else:
                offset_braid.append(Generator(g - 3*i))
    return Braid(offset_braid), expval

# Generate a brickwall braid on `strands` strands with depth at most twice `layers`.
# If a braidword `center` is not provided, the generated braidword will be equivalent
# to the the trivial braid. Otherwise, the Markov closure of the generated braid will 
# be equivalent to that of `center`, extended with unlinks up to `strands` strands.
def brickwall_unlink(strands: int, layers: int, center: Braid = None, prob: float = 0.666) -> tuple[Braid, tuple[int, int]]:
    # The Z-position of the strand at each Y-position
    z = [i for i in range(strands)]
    random.shuffle(z)
    # The original Y-position of the strand at each Y-position
    p = [i for i in range(strands)]

    gs = []
    # Generate a brickwall circuit as above:
    for l in range(layers):
        for idx in range(l % 2, strands - 1, 2):
            # If we want to place a generator:
            if random.random() < prob:
                # Use the Z-positions to ensure that strands
                # are properly layered and never cross
                if z[idx] < z[idx+1]:
                    gs.append(Generator(idx + 1))
                else:
                    gs.append(Generator(-(idx + 1)))
                # Update the original Y-position and Z-position of
                # the strands after this generator
                z[idx], z[idx+1] = z[idx+1], z[idx]
                p[idx], p[idx+1] = p[idx+1], p[idx]
    
    # If we want, can insert some generators here, and then the overall braid
    # will be equivalent to the braid given by the braidword `center` extended
    # to be as wide as `strands`, up to a permutation (in particular, the 
    # Markov closure will be the same)
    cstart = len(gs)
    if center is not None:
        for g in center.gens:
            gs.append(g)
    cend = len(gs)

    # Now generate the inverse of the circuit above using a brickwall sorting 
    # network. It is a neat result that any permutation generated by brickwall
    # can be sorted by a brickwall sorting network of the same number of layers + 1
    for l in range(layers + 1):
        for idx in range(l % 2, strands - 1, 2):
            # If two strands are in the wrong relative order, swap them
            if p[idx] > p[idx+1]:
                # Use the Z-positions to ensure correct layering
                if z[idx] < z[idx+1]:
                    gs.append(Generator(idx + 1))
                else:
                    gs.append(Generator(-(idx + 1)))
                # Update the Z- and original Y-positions
                z[idx], z[idx+1] = z[idx+1], z[idx]
                p[idx], p[idx+1] = p[idx+1], p[idx]
    # At this point `p` should be sorted. 
    assert sorted(p) == p

    return Braid(gs), (cstart, cend)

def create_braid(args: argparse.Namespace) -> tuple[str, list[tuple[dict, complex | None, Braid]]]:
    if args.word is not None:
        word = [int(i) for i in args.word.split()]
        name = 'word' + '_'.join(map(str, word))
        return name, [({ "method": "word", "word": word }, None, Braid.from_word(word))]
    elif args.planar is not None:
        import spherogram

        ints = [int(i) for i in args.planar.split()]
        code = list(zip(ints[0::4], ints[1::4], ints[2::4], ints[3::4]))
        link = spherogram.Link(code)
        name = 'pd' + '_'.join(map(str, ints))
        return name, [({ "method": "pd", "pd": code }, None, Braid.from_word(link.braid_word()))]
    elif args.dt_code is not None:
        import spherogram

        ints = [int(i) for i in args.dt_code.split()]
        link = spherogram.Link(f'DT: [({",".join(map(str, ints))})]')
        name = 'dt' + '_'.join(map(str, ints))
        return name, [({ "method": "dt", "dt": ints }, None, Braid.from_word(link.braid_word()))]
    elif args.name is not None:
        import spherogram

        link = spherogram.Link(args.name.strip())
        return args.name.strip(), [({ "method": "name", "name": args.name }, None, Braid.from_word(link.braid_word()))]

def random_braid(args: argparse.Namespace) -> tuple[str, list[tuple[dict, complex | None, Braid]]]:
    gen = {
        "method": "random",
        "strands": args.strands,
        "layers": args.layers,
        "prob": args.prob,
        "kind": "brickwall" if not args.uniform else "uniform"      
    }
    return f"r_n{args.strands}_l{args.layers}", [
        (gen, None, brickwall(args.strands, args.layers, args.prob)) if not args.uniform else (gen, None, uniform(args.strands, args.layers))
        for _ in range(args.count)
    ]

def benchmark_braid(args: argparse.Namespace) -> tuple[str, list[tuple[dict, complex | None, Braid]]]:
    source = {
        "method": "augment",
        "strands": args.strands,
        "layers": args.layers,
        "prob": args.prob,
    }
    output = []
    for _ in tqdm.trange(args.count, desc="generating benchmark braids", leave=False):
        if args.inner:
            center, expval = random3_layer(args.strands)
        else:
            center, expval = Braid([]), 1.0
        while True:
            braid, cpos = brickwall_unlink(args.strands, args.layers, center=center, prob=args.prob)
            if braid.strands == args.strands:
                break
        jones = (-np.exp(-1j*3*np.pi/5))**(3*braid.writhe) * phi ** (braid.strands - 1) * expval
        sc = source.copy()
        sc['center'] = {
            'word': [c.idx for c in center],
            'pos': cpos
        }
        output.append((sc, (jones.real, jones.imag), braid))
    return f"b_n{braid.strands}_l{args.layers}", output

if __name__ == "__main__":
    if args.command == 'create':
        name, braids = create_braid(args)
    elif args.command == 'random':
        name, braids = random_braid(args)
    elif args.command == 'benchmark':
        name, braids = benchmark_braid(args)

    optimized: list[tuple[float, Braid]] = []
    for source, jones, braid in tqdm.tqdm(braids, desc="minimizing braids", leave=False):
        before = time.time()
        if args.minimize == 'none':
            pass
        elif args.minimize == 'cancel':
            braid = minimize.cancel_inverses(braid, cyclic=not args.non_cyclic)
        elif args.minimize == 'peephole':
            braid = minimize.peephole_minimize(braid, cyclic=not args.non_cyclic)
        elif args.minimize == 'full':
            braid = minimize.minimize(braid, n=args.steps, cyclic=not args.non_cyclic)
        after = time.time()
        optimized.append((after - before, braid))

    costs = []
    for idx, ((source, jones, braid), (otime, obraid)) in enumerate(zip(tqdm.tqdm(braids, desc="writing output files", leave=False), optimized)):
        costs.append(minimize.cost_function(obraid))
        val = {
            "optimized": {
                "word": obraid.to_word(),
                "cost": minimize.cost_function(obraid),
                "time": otime,
                "method": args.minimize,
                "steps": args.steps,
                "cyclic": not args.non_cyclic
            },
            "original": {
                "word": braid.to_word(),
                "cost": minimize.cost_function(braid)
            },
            "source": source,
            "jones": jones
        }

        if len(braids) == 1:
            fname: pathlib.Path = pathlib.Path(args.out_dir) / (name + '.json')
        else:
            fname: pathlib.Path = pathlib.Path(args.out_dir) / (name + f'_c{idx}.json')
        
        with open(fname, "w") as f:
            json.dump(val, f, indent=4)

    print(f"wrote {len(costs)} files to {pathlib.Path(args.out_dir)}. min cost = {min(costs)}, max cost = {max(costs)}, avg cost = {sum(costs) / len(costs):.2f}")
