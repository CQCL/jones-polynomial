import math
import random

from common import Braid, Generator

# Find the leftmost reducible handle in a braidword
def find_handle(braid: Braid) -> tuple[int, int] | None:
    for q in range(1, len(braid)):
        for k in range(q - 1, -1, -1):
            if braid[k].idx != -braid[q].idx:
                continue
            
            j = braid[k].pos
            for l in braid.gens[k+1:q]:
                if j - 1 <= l.pos <= j:
                    break
            else:
                return (k, q)

# Given the location of a handle in a braidword, reduce the handle
def remove_handle(braid: Braid, p: int, q: int):
    ngens = []
    for gen in braid.gens[p:q+1]:
        if gen.pos != braid[p].pos and gen.pos != braid[p].pos + 1:
            ngens.append(gen)
        elif gen.pos == braid[p].pos + 1:
            ngens.append(Generator(-braid[p].sign*gen.pos))
            ngens.append(Generator(gen.sign*braid[p].pos))
            ngens.append(Generator(braid[p].sign*gen.pos))
    braid.gens[p:q+1] = ngens

# Apply Dehornoy handle reduction to reduce the braid
def handle_reduction(braid: Braid):
    while True:
        handle = find_handle(braid)
        if handle is None:
            break
        p, q = handle
        remove_handle(braid, p, q)

# Compute the Dynnikov coordinates of a braid word,
# these uniquely characterize the braid represented by this word
def dynnikov_coordinates(braid: Braid) -> list[int]:
    coords = [0, 1] * braid.strands
    for g in braid.gens:
        i = g.pos - 1
        x1, y1, x2, y2 = coords[2*i:2*i+4]
        if g.sign > 0:
            z = x1 - min(y1, 0) - x2 + max(y2, 0)
            coords[2*i+0] = x1 + max(y1, 0) + max(max(y2, 0) - z, 0)
            coords[2*i+1] = y2 - max(z, 0)
            coords[2*i+2] = x2 + min(y2, 0) + min(min(y1, 0) + z, 0)
            coords[2*i+3] = y1 + max(z, 0)
        else:
            z = x1 + min(y1, 0) - x2 - max(y2, 0)
            coords[2*i+0] = x1 - max(y1, 0) - max(max(y2, 0) + z, 0)
            coords[2*i+1] = y2 + min(z, 0)
            coords[2*i+2] = x2 - min(y2, 0) - min(min(y1, 0) - z, 0)
            coords[2*i+3] = y1 - min(z, 0)
    return coords

# Check if two braidwords represent the same braid
def compare(b1: Braid, b2: Braid) -> bool:
    c1 = dynnikov_coordinates(b1)
    c2 = dynnikov_coordinates(b2)

    # If the braids don't have the same number of strands
    # add extra coordinates to the smaller braid
    if len(c1) != len(c2):
        while len(c1) < len(c2):
            c1 += [0, 1]
        while len(c2) < len(c1):
            c2 += [0, 1]
    
    return c1 == c2

# Check if a braidword represents the trivial braid
def is_trivial(b: Braid) -> bool:
    c = dynnikov_coordinates(b)
    return all(cc == i % 2 for i, cc in enumerate(c))

# Step one of 3-strand minimization, we remove all occurences
# of 121 and 212 (and their inverses) by pulling them through
def clear_twists(braid: Braid) -> int:
    p = 0
    while True:
        for i in range(len(braid) - 2):
            if braid[i].pos == braid[i+1].pos \
                or braid[i+1].pos == braid[i+2].pos \
                or braid[i].sign != braid[i+1].sign \
                or braid[i+1].sign != braid[i+2].sign:
                continue
            
            # Put the twist at the front
            if braid[i].idx < 0:
                p -= 1
            else:
                p += 1

            # Delete the twist
            del braid[i+2]
            del braid[i+1]
            del braid[i]

            # Flip everything in front of it
            for j in range(i):
                if braid[j].sign == -1:
                    braid[j].idx = -3 - braid[j].idx
                else:
                    braid[j].idx = 3 - braid[j].idx
            break
        else:
            break
    return p

# Step two of 3-strand minimization, we remove all
# occurences of 12 and 21 and their inverses by introducing twists
def clear_wraps(braid: Braid) -> int:
    p = 0
    while True:
        for i in range(len(braid) - 1):
            if braid[i].pos == braid[i+1].pos or braid[i].sign != braid[i+1].sign:
                continue
            
            # Introduce a generator and its inverse to make the 
            # wrap a twist, then pull the twist to the front
            if braid[i].sign == -1:
                p -= 1
            else:
                p += 1
            del braid[i+1]
            braid[i].idx *= -1

            # Flip everything in front of it
            for j in range(i):
                if braid[j].sign == -1:
                    braid[j].idx = -3 - braid[j].idx
                else:
                    braid[j].idx = 3 - braid[j].idx
            break
        else:
            break
    return p

# Step three of 3-strand minimization, we put as many
# twists as we can back into the braid to form wraps
def place_wraps(braid: Braid, p: int) -> int:
    while p != 0:
        for i in range(len(braid)):
            # We can place a wrap wherever there
            # is a generator of opposite sign to the twist
            if p * braid[i].sign > 0:
                continue
            
            # Place the twist at this point,
            # one of the generators cancels
            braid[i].idx *= -1
            if braid[i].sign == 1:
                p -= 1
                braid.gens.insert(i + 1, Generator(3 - braid[i].idx))
            else:
                p += 1
                braid.gens.insert(i + 1, Generator(-3 - braid[i].idx))

            # Flip everything up to the new twist
            for j in range(i):
                if braid[j].sign == -1:
                    braid[j].idx = -3 - braid[j].idx
                else:
                    braid[j].idx = 3 - braid[j].idx
            break
        else:
            break
    return p

# Implements Berger's algorithm for minimizing 3-stranded braids
# in a way that tries to maximize consecutive generators
def minimize3(braid: Braid) -> Braid:
    assert braid.strands <= 3

    braid = braid.copy()
    
    # Cancel inverses where possible
    while True:
        for i in range(len(braid)-1):
            if braid[i+1].idx == -braid[i].idx:
                del braid[i+1]
                del braid[i]
                break
        else:
            break
    
    # Do Berger's algorithm
    p = clear_twists(braid)
    p += clear_wraps(braid)
    p = place_wraps(braid, p)

    # If p = 0 there are no twists left over
    if p == 0:
        return braid

    # Place twists at the start in such a way
    # that there is two consective generators at the
    # start of the braid whenever possible, and 
    # the same between each pair of twists
    idx = braid[0].pos if len(braid) > 0 else 1
    if p < 0:
        if idx == 1:
            return Braid([Generator(-1), Generator(-2), Generator(-1)] * (-p) + braid.gens)
        else:
            return Braid([Generator(-2), Generator(-1), Generator(-2)] * (-p) + braid.gens)
    else:
        if idx == 1:
            return Braid([Generator(1), Generator(2), Generator(1)] * (p) + braid.gens)
        else:
            return Braid([Generator(2), Generator(1), Generator(2)] * (p) + braid.gens)

# Find all subwords of a braid that act only on three strands
def find_segments(braid: Braid, cyclic: bool = True) -> list[tuple[list[int], int]]:
    segments = []
    for i in range(len(braid)):
        segment = [i]
        g = braid[i].pos
        # The other generator in this 3-strand subword, initially none
        o = None
        limit = len(braid) + i if cyclic else len(braid)
        for j in range(i+1, limit):
            j = j % len(braid)
            pos = braid[j].pos
            # If it commutes with both generators, skip
            if abs(pos - g) > 1 and (o is None or abs(pos - o) > 1):
                continue
            
            # If there is no other generator yet
            # and this is distinct, then this is it
            if o is None and pos != g:
                o = pos

            # If this generator is on the right strands, add it
            if pos == o or pos == g:
                segment.append(j)
            else:
                break
        # Find the offset needed to transform this to 1, 2
        idx = g - 1 if o is None else min(g, o) - 1
        segments.append((segment, idx))
    return segments

# Given a list of locations that can be made adjacent, make them adjacent
def condense_gens(braid: Braid, indices: list[int]) -> list[int]:
    indices = list(indices)

    if len(indices) == 1:
        return indices

    # First, cycle the braid so that indices is strictly increasing
    if indices[-1] != max(indices):
        off = indices[-1]+1
        braid.gens = braid.gens[off:] + braid.gens[:off]
        indices = [
            idx - off if idx > indices[-1] else idx - off + len(braid.gens) for idx in indices
        ]

    # Shuffle everything towards their right neighbor
    for i in range(len(indices)-2, -1, -1):
        while indices[i]+1 < indices[i+1]:
            if not braid[indices[i]].overlap(braid[indices[i]+1]):
                braid[indices[i]], braid[indices[i]+1] = braid[indices[i]+1], braid[indices[i]]
                indices[i] += 1
            else:
                break

    # Shuffle everything towards their left neighbor
    for i in range(1, len(indices)):
        while indices[i]-1 > indices[i-1]:
            if not braid[indices[i]].overlap(braid[indices[i]-1]):
                braid[indices[i]], braid[indices[i]-1] = braid[indices[i]-1], braid[indices[i]]
                indices[i] -= 1
            else:
                break

    return indices

# Optimize a braid by minimizing the longest 3-stranded
# subword using Berger's algorithm until we stop improving
def peephole_minimize(braid: Braid, cyclic: bool = True) -> Braid:
    braid = braid.copy()
    while True:
        # Sort 3-stranded subwords by length
        segments = find_segments(braid, cyclic)
        segments.sort(key = lambda s: len(s[0]), reverse=True)

        for segment, shift in segments:
            # Try minimizing each subword
            orig = Braid.from_word([braid[i].idx - shift if braid[i].sign > 0 else braid[i].idx + shift for i in segment])
            min = minimize3(orig)
            if len(min) < len(orig):
                # If this improved things, update the braid
                segment = condense_gens(braid, segment)
                p, q = segment[0], segment[-1] + 1
                braid.gens[p:q] = [Generator(m.idx + shift) if m.sign > 0 else Generator(m.idx - shift) for m in min.gens]
                break
        else:
            # If none of them helped, stop here
            break
    return braid

# Find a random occurence of 121 or 212 (or its inverse) in a braid
def find_random_slide(braid: Braid, cyclic: bool = True) -> tuple[int, int, int] | None:
    # Iterate over potential start points in a random order
    choices = list(range(len(braid) if cyclic else len(braid)-2))
    random.shuffle(choices)

    for i in choices:
        # Find one subsequent generator that does not commute
        limit = len(braid)+i-1 if cyclic else len(braid) - 1
        for j in range(i+1, limit):
            if abs(braid[i%len(braid)].pos - braid[j%len(braid)].pos) <= 1:
                break
        else:
            continue
        
        # Now find another one after the first
        limit = len(braid)+i if cyclic else len(braid)
        for k in range(j+1, limit):
            if abs(braid[i%len(braid)].pos - braid[k%len(braid)].pos) <= 1:
                break
        else:
            continue

        i %= len(braid)
        j %= len(braid)
        k %= len(braid)
        
        # If this looks like a slide we are done
        if braid[i].idx == braid[k].idx and braid[i].sign == braid[j].sign:
            return (i, j, k)
        
# Minimize a braid by peephole minimization and random slides 
# a fixed number of times or until there are no more slides
def random_minimize(braid: Braid, n: int, cyclic: bool = True) -> Braid:
    braid = peephole_minimize(braid, cyclic)
    for _ in range(n):
        s = find_random_slide(braid, cyclic)
        if s is not None:
            i, j, k = condense_gens(braid, s)
            braid[i].idx, braid[j].idx, braid[k].idx = braid[j].idx, braid[i].idx, braid[j].idx
            braid = peephole_minimize(braid, cyclic)
        else:
            break
    return braid   

# Try to randomize a braid as much as possible
def randomize(braid: Braid, n: int = 500, cyclic: bool = True) -> Braid:
    braid = braid.copy()
    for _ in range(n):
        if cyclic:
            shift = random.randrange(len(braid))
            braid.gens = braid.gens[shift:] + braid.gens[:shift]

        s = find_random_slide(braid, cyclic)
        if s is not None:
            i, j, k = condense_gens(braid, s)
            braid[i].idx, braid[j].idx, braid[k].idx = braid[j].idx, braid[i].idx, braid[j].idx
        else:
            break

        segments = find_segments(braid, False)
        indices = list(range(len(segments)))
        random.shuffle(indices)

        for idx in indices:
            segment, shift = segments[idx]
            orig = Braid.from_word([braid[i].idx - shift if braid[i].sign > 0 else braid[i].idx + shift for i in segment])
            min = minimize3(orig)

            if any(a.idx != b.idx for a, b in zip(orig.gens, min.gens)):
                segment = condense_gens(braid, segment)
                p, q = segment[0], segment[-1] + 1
                braid.gens[p:q] = [Generator(m.idx + shift) if m.sign > 0 else Generator(m.idx - shift) for m in min.gens]
                break
                    
    return braid
        

def cancel_inverses(braid: Braid, cyclic: bool = True) -> Braid:
    braid = braid.copy()
    
    # Cancel inverses where possible
    while True:
        limit = len(braid) if cyclic else len(braid) - 1
        for i in range(limit):
            # Find a subsequent generator that does not commute
            limit = len(braid)+i if cyclic else len(braid)
            for j in range(i+1, limit):
                if abs(braid[i].pos - braid[j%len(braid)].pos) <= 1:
                    break
            else:
                continue
            j %= len(braid)
            
            # If this is a pair of inverses, delete them
            if braid[j].idx == -braid[i].idx:
                if i < j:
                    del braid[j]
                    del braid[i]
                else:
                    del braid[i]
                    del braid[j]
                break
        else:
            break

    return braid

# Find all runs of consecutive identical generators in a braid
def find_consecutive(braid: Braid) -> list[list[int]]:
    marked = set()
    runs = []
    for i in range(len(braid)):
        if i in marked:
            continue
        run = [i]
        marked.add(i)
        for j in range(i+1, len(braid)):
            if abs(braid[i].pos - braid[j].pos) > 1:
                continue

            if braid[i].idx == braid[j].idx:
                run.append(j)
                marked.add(j)
            else:
                break
        runs.append(run)
    return runs

def cost_function(braid: Braid) -> int:
    runs = find_consecutive(braid)
    return len(runs)

# Use simulated annealing on slide moves to try
# and increase the number of consecutive generators
def simulated_annealing(braid: Braid, start: float, end: float, steps: float) -> Braid:
    braid = braid.copy()
    score = cost_function(braid)
    best_braid = braid.copy()
    best_score = score
    for st in range(steps):
        s = find_random_slide(braid)
        if s is None:
            break
        i, j, k = condense_gens(braid, s)
        braid[i].idx, braid[j].idx, braid[k].idx = braid[j].idx, braid[i].idx, braid[j].idx
        new_score = cost_function(braid)
        temp = start * (end / start) ** (st / (steps-1))
        if new_score <= score or random.random() < math.exp(-(new_score - score)/temp):
            score = new_score
        else:
            braid[i].idx, braid[j].idx, braid[k].idx = braid[j].idx, braid[i].idx, braid[j].idx
        if new_score < best_score:
            best_score = new_score
            best_braid = braid.copy()
    return best_braid

# Minimize a closed braid as much as possible
def minimize(b: Braid, n: int = 100, cyclic: bool = True) -> Braid:
    return random_minimize(cancel_inverses(b, cyclic), n, cyclic)
