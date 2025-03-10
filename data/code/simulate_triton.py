import argparse
import rich_argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=rich_argparse.RichHelpFormatter)
    parser.add_argument("SPEC", help="machine specification file")
    parser.add_argument("SHOTS", help="number of shots per classical bit string input", type=int)
    parser.add_argument("BRAID", help="braid json file(s)", nargs="+")
    parser.add_argument("-g", "--gpu", help="which cuda device to use for simulation", type=int, default=0)
    parser.add_argument("-b", "--bits", help="2D npy arrays containing classical bit string inputs", nargs="+")
    parser.add_argument("-o", "--output", help="name of measurement output file", nargs="+")
    parser.add_argument("-s", "--scale", help="scale noise rate by a constant factor", type=float, default=1.0)
    parser.add_argument("-B", "--batchsize", help="maximum number of classical bitstrings per batch", type=int, default=256)
    parser.add_argument("-t", "--time-limit", help="set a maximum time limit after which the program will exit", type=float)
    parser.add_argument("-z", "--zero-angles", help="set all gate angles in crossings to zero, to measure memory errors", action='store_true')
    parser.add_argument("-m", "--memory-error", help="coherent dephasing memory error scale", type=float)
    parser.add_argument("-c", "--conjugate", help="simulate the conjugate of the input braid", action='store_true')
    args = parser.parse_args()

from common import Braid
import minimize
import numpy as np
import json
import cmath
import string
import pathlib
import tqdm
import torch
import triton
import triton.language as tl
import dataclasses
import time

def get_generator_power(spec: dict, power: int, offset: int, scale: float, nidx: int):
    power %= 10
    if power == 0:
        return 0.0, [], [], nidx

    op = [(('crossing', [power - 1], [offset, offset+1, offset+2], None, None, nidx))]

    pidx = 0
    nprob = []
    for name in ["Rz", "Rz", "Rz", "U1q", "Rz", "RZZ", "U1q", "Rz", "Rz", "RZZ", "Rz", "U1q", "Rz", "RZZ", "U1q", "Rz", "Rz"]:
        nparams = 2 if name == "U1q" else 1
        nqubits = 2 if name == "RZZ" else 1
        noise = get_noise(spec, name, crossing_angles[power - 1][pidx:pidx+nparams], scale)
        if noise is not None:
            if nqubits == 1:
                nprob.append((noise, 2))
                nidx += 2
            elif nqubits == 2:
                nprob.append((noise, 4))
                nidx += 4
        pidx += nparams

    return eigenphases[power - 1], op, nprob, nidx

def get_braid_unitary(spec: dict, braid: Braid, scale: float, nidx: int):
    ops = []
    eigenphase = 0.0
    nprobs = []
    depths = [0]*(braid.strands + 1)
    mprobs = []
    for run in minimize.find_consecutive(braid):
        power = braid[run[0]].sign * len(run)
        offset = braid[run[0]].pos - 1
        phase, op, nprob, nidx = get_generator_power(spec, power, offset, scale, nidx)
        eigenphase += phase
        ops.extend(op)
        nprobs.extend(nprob)
        maxdepth = max(depths)
        mprobs.append((maxdepth - depths[offset + 0], maxdepth - depths[offset + 1], maxdepth - depths[offset + 2]))
        depths[offset + 0] = maxdepth + 1
        depths[offset + 1] = maxdepth + 1
        depths[offset + 2] = maxdepth + 1
    eigenphase %= (2*cmath.pi)

    return eigenphase, ops, nprobs, nidx, mprobs

def construct_cfev(spec: dict, braid: Braid, scale: float):
    ops = []
    nqubits = braid.strands + 1
    nprobs = []
    nidx = 0
    
    # Apply an H gate to qubit 1
    ops.append(("U1q", [cmath.pi/2.0, -cmath.pi/2.0], [1], None, None, nidx))
    nprobs.append((get_noise(spec, 'U1q', [cmath.pi/2.0, -cmath.pi/2.0], scale), 2))
    nidx += 2
    ops.append(("Rz", [cmath.pi], [1], None, None, None))

    # Apply an S^+ gate to qubit 1 conditioned on cbit 0
    ops.append(("Rz", [-cmath.pi/2.0], [1], 0, 1, None))

    # Apply V_cat
    for i in range(2, nqubits):
        ops.append(('cnot', [], [1, i], i-1, 1, nidx))
        nprobs.append((get_noise(spec, 'U1q', [-cmath.pi/2.0, cmath.pi/2.0], scale), 2))
        nprobs.append((get_noise(spec, 'RZZ', [cmath.pi/2.0], scale), 4))
        nprobs.append((get_noise(spec, 'U1q', [cmath.pi/2.0, cmath.pi], scale), 2))
        nidx += 8

    # Apply the braid unitary
    eigenphase, braid_ops, braid_nprobs, nidx, mprobs = get_braid_unitary(spec, braid, scale, nidx)
    ops.extend(braid_ops)
    nprobs.extend(braid_nprobs)

    # Apply V_cat^+
    for i in range(nqubits - 1, 2 - 1, -1):
        ops.append(('cnot', [], [1, i], i-1, 1, nidx))
        nprobs.append((get_noise(spec, 'U1q', [-cmath.pi/2.0, cmath.pi/2.0], scale), 2))
        nprobs.append((get_noise(spec, 'RZZ', [cmath.pi/2.0], scale), 4))
        nprobs.append((get_noise(spec, 'U1q', [cmath.pi/2.0, cmath.pi], scale), 2))
        nidx += 8

    # Apply Rz(eigenphase) to qubit 1
    ops.append(("Rz", [eigenphase], [1], None, None, None))

    # Apply an H gate to qubit 1
    ops.append(("U1q", [cmath.pi/2.0, -cmath.pi/2.0], [1], None, None, nidx))
    nprobs.append((get_noise(spec, 'U1q', [cmath.pi/2.0, -cmath.pi/2.0], scale), 2))
    nidx += 2
    ops.append(("Rz", [cmath.pi], [1], None, None, None))
    
    if spec["spam"]["prepare"] != 0.0:
        for _ in range(nqubits):
            nprobs.append((spec["spam"]["prepare"], 1))

    return ops, nqubits, nprobs, mprobs

# Get the Pauli channel corresponding to the noise for
# a given native gate, or None if there is no noise.
def get_noise(spec: dict, gate: str, params: "list[float]", scale: float) -> float | None: 
    if "error" not in spec["gates"][gate]:
        return
    
    rate = spec["gates"][gate]["error"]["rate"]
    channel = spec["gates"][gate]["error"]["channel"]

    if isinstance(rate, float):
        pass
    elif isinstance(rate, dict):
        # Rate may depend on the parameters, in which case
        # we will have a dictionary with code to eval
        params = { l: p for l, p in zip(string.ascii_lowercase, params) }
        rate = eval(rate["code"], cmath.__dict__.copy(), params)
    else:
        raise ValueError()

    rate *= scale

    if channel != "depolarizing":
        raise ValueError("Only depolarizing error channels are supported")

    return rate

@triton.jit
def ctrl_pauli_kernel(sv_r_ptr, sv_i_ptr, ctrl_z_ptr, ctrl_x_ptr, width, nbatch, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(axis=1)
    elem_id = tl.program_id(axis=0)
    
    x = elem_id >> width
    y = elem_id & ((1 << width) - 1)
    block_offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_offsets < nbatch
    offsets_0 = nbatch * (((2 * x + 0) << width) + y) + block_offsets
    offsets_1 = nbatch * (((2 * x + 1) << width) + y) + block_offsets

    ctrl_z = tl.load(ctrl_z_ptr + block_offsets, mask=mask)
    ctrl_x = tl.load(ctrl_x_ptr + block_offsets, mask=mask)
    in_0_r = tl.load(sv_r_ptr + offsets_0, mask=mask)
    in_0_i = tl.load(sv_i_ptr + offsets_0, mask=mask)
    in_1_r = tl.load(sv_r_ptr + offsets_1, mask=mask)
    in_1_i = tl.load(sv_i_ptr + offsets_1, mask=mask)

    out_0_r = (1.0 - ctrl_x) * in_0_r + ctrl_x * (1.0 - 2.0 * ctrl_z) * in_1_r
    out_0_i = (1.0 - ctrl_x) * in_0_i + ctrl_x * (1.0 - 2.0 * ctrl_z) * in_1_i
    out_1_r = ctrl_x * in_0_r + (1.0 - ctrl_x) * (1.0 - 2.0 * ctrl_z) * in_1_r
    out_1_i = ctrl_x * in_0_i + (1.0 - ctrl_x) * (1.0 - 2.0 * ctrl_z) * in_1_i

    tl.store(sv_r_ptr + offsets_0, out_0_r, mask=mask)
    tl.store(sv_i_ptr + offsets_0, out_0_i, mask=mask)
    tl.store(sv_r_ptr + offsets_1, out_1_r, mask=mask)
    tl.store(sv_i_ptr + offsets_1, out_1_i, mask=mask)

@triton.jit
def ctrl_rz_kernel(sv_r_ptr, sv_i_ptr, ctrl_ptr, width, nbatch, theta, BLOCK_SIZE: tl.constexpr):
    s = tl.sin(theta)
    c = tl.cos(theta)
    
    block_id = tl.program_id(axis=1)
    elem_id = tl.program_id(axis=0)
    
    x = elem_id >> width
    y = elem_id & ((1 << width) - 1)
    block_offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_offsets < nbatch
    offsets_1 = nbatch * (((2 * x + 1) << width) + y) + block_offsets

    ctrl = tl.load(ctrl_ptr + block_offsets, mask=mask)
    in_1_r = tl.load(sv_r_ptr + offsets_1, mask=mask)
    in_1_i = tl.load(sv_i_ptr + offsets_1, mask=mask)

    cc = 1.0 + ctrl * (c - 1.0)
    cs = ctrl * s
    out_1_r = cc * in_1_r - cs * in_1_i
    out_1_i = cs * in_1_r + cc * in_1_i

    tl.store(sv_r_ptr + offsets_1, out_1_r, mask=mask)
    tl.store(sv_i_ptr + offsets_1, out_1_i, mask=mask)

@triton.jit
def var_rz_kernel(sv_r_ptr, sv_i_ptr, theta_ptr, width, nbatch, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(axis=1)
    elem_id = tl.program_id(axis=0)
    
    x = elem_id >> width
    y = elem_id & ((1 << width) - 1)
    block_offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_offsets < nbatch
    offsets_1 = nbatch * (((2 * x + 1) << width) + y) + block_offsets

    theta = tl.load(theta_ptr + block_offsets, mask=mask)
    s = tl.sin(theta)
    c = tl.cos(theta)

    in_1_r = tl.load(sv_r_ptr + offsets_1, mask=mask)
    in_1_i = tl.load(sv_i_ptr + offsets_1, mask=mask)

    out_1_r = c * in_1_r - s * in_1_i
    out_1_i = s * in_1_r + c * in_1_i

    tl.store(sv_r_ptr + offsets_1, out_1_r, mask=mask)
    tl.store(sv_i_ptr + offsets_1, out_1_i, mask=mask)

@triton.jit
def ctrl_u1q_kernel(sv_r_ptr, sv_i_ptr, ctrl_ptr, noise_x_ptr, noise_z_ptr, width, nbatch, theta, phi, BLOCK_SIZE: tl.constexpr):
    ct2 = tl.cos(theta / 2.0)
    st2 = tl.sin(theta / 2.0)
    cp = tl.cos(phi)
    sp = tl.sin(phi)

    block_id = tl.program_id(axis=1)
    elem_id = tl.program_id(axis=0)
    
    x = elem_id >> width
    y = elem_id & ((1 << width) - 1)
    block_offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_offsets < nbatch
    offsets_0 = nbatch * (((2 * x + 0) << width) + y) + block_offsets
    offsets_1 = nbatch * (((2 * x + 1) << width) + y) + block_offsets

    ctrl = tl.load(ctrl_ptr + block_offsets, mask=mask)
    noise_x = tl.load(noise_x_ptr + block_offsets, mask=mask)
    noise_z = tl.load(noise_z_ptr + block_offsets, mask=mask)
    in_0_r = tl.load(sv_r_ptr + offsets_0, mask=mask)
    in_0_i = tl.load(sv_i_ptr + offsets_0, mask=mask)
    in_1_r = tl.load(sv_r_ptr + offsets_1, mask=mask)
    in_1_i = tl.load(sv_i_ptr + offsets_1, mask=mask)

    im_0_r = ct2 * in_0_r + st2 * (cp * in_1_i - sp * in_1_r)
    im_0_i = ct2 * in_0_i + st2 * (-sp * in_1_i - cp * in_1_r)
    im_1_r = ct2 * in_1_r + st2 * (cp * in_0_i + sp * in_0_r)
    im_1_i = ct2 * in_1_i + st2 * (sp * in_0_i - cp * in_0_r)
    n_0_r = (1.0 - noise_x) * im_0_r + noise_x * (1.0 - 2.0 * noise_z) * im_1_r
    n_0_i = (1.0 - noise_x) * im_0_i + noise_x * (1.0 - 2.0 * noise_z) * im_1_i
    n_1_r = noise_x * im_0_r + (1.0 - noise_x) * (1.0 - 2.0 * noise_z) * im_1_r
    n_1_i = noise_x * im_0_i + (1.0 - noise_x) * (1.0 - 2.0 * noise_z) * im_1_i
    out_0_r = (1.0 - ctrl) * in_0_r + ctrl * n_0_r
    out_0_i = (1.0 - ctrl) * in_0_i + ctrl * n_0_i
    out_1_r = (1.0 - ctrl) * in_1_r + ctrl * n_1_r
    out_1_i = (1.0 - ctrl) * in_1_i + ctrl * n_1_i

    tl.store(sv_r_ptr + offsets_0, out_0_r, mask=mask)
    tl.store(sv_i_ptr + offsets_0, out_0_i, mask=mask)
    tl.store(sv_r_ptr + offsets_1, out_1_r, mask=mask)
    tl.store(sv_i_ptr + offsets_1, out_1_i, mask=mask)

@triton.jit
def rz_subkernel(in_1_r, in_1_i, angles_ptr, THETA: tl.constexpr):
    theta = tl.load(angles_ptr + THETA)
    c = tl.cos(theta)
    s = tl.sin(theta)
    out_1_r = c * in_1_r - s * in_1_i
    out_1_i = s * in_1_r + c * in_1_i
    return out_1_r, out_1_i

@triton.jit
def u1q_subkernel(in_0_r, in_0_i, in_1_r, in_1_i, noise_x_ptr, noise_z_ptr, block_offsets, mask, angles_ptr, THETA: tl.constexpr, PHI: tl.constexpr):
    theta = tl.load(angles_ptr + THETA)
    phi = tl.load(angles_ptr + PHI)
    ct2 = tl.cos(theta / 2.0)
    st2 = tl.sin(theta / 2.0)
    cp = tl.cos(phi)
    sp = tl.sin(phi)

    noise_x = tl.load(noise_x_ptr + block_offsets, mask=mask)
    noise_z = tl.load(noise_z_ptr + block_offsets, mask=mask)
    im_0_r = ct2 * in_0_r + st2 * (cp * in_1_i - sp * in_1_r)
    im_0_i = ct2 * in_0_i + st2 * (-sp * in_1_i - cp * in_1_r)
    im_1_r = ct2 * in_1_r + st2 * (cp * in_0_i + sp * in_0_r)
    im_1_i = ct2 * in_1_i + st2 * (sp * in_0_i - cp * in_0_r)
    n_0_r = (1.0 - noise_x) * im_0_r + noise_x * (1.0 - 2.0 * noise_z) * im_1_r
    n_0_i = (1.0 - noise_x) * im_0_i + noise_x * (1.0 - 2.0 * noise_z) * im_1_i
    n_1_r = noise_x * im_0_r + (1.0 - noise_x) * (1.0 - 2.0 * noise_z) * im_1_r
    n_1_i = noise_x * im_0_i + (1.0 - noise_x) * (1.0 - 2.0 * noise_z) * im_1_i
    return n_0_r, n_0_i, n_1_r, n_1_i

@triton.jit
def rzz_subkernel(
    in_00_r, in_00_i, in_01_r, in_01_i, in_10_r, in_10_i, in_11_r, in_11_i,
    noise_1_x_ptr, noise_1_z_ptr, noise_2_x_ptr, noise_2_z_ptr, block_offsets, mask,
    angles_ptr, THETA: tl.constexpr
):
    theta = tl.load(angles_ptr + THETA)
    s = tl.sin(theta)
    c = tl.cos(theta)

    noise_1_x = tl.load(noise_1_x_ptr + block_offsets, mask=mask)
    noise_1_z = tl.load(noise_1_z_ptr + block_offsets, mask=mask)
    noise_2_x = tl.load(noise_2_x_ptr + block_offsets, mask=mask)
    noise_2_z = tl.load(noise_2_z_ptr + block_offsets, mask=mask)

    im_01_r = c * in_01_r - s * in_01_i
    im_01_i = s * in_01_r + c * in_01_i
    im_10_r = c * in_10_r - s * in_10_i
    im_10_i = s * in_10_r + c * in_10_i
    im_00_r = in_00_r
    im_00_i = in_00_i
    im_11_r = in_11_r
    im_11_i = in_11_i

    nz_00_r = im_00_r
    nz_00_i = im_00_i
    nz_01_r = (1.0 - 2.0 * noise_2_z) * im_01_r
    nz_01_i = (1.0 - 2.0 * noise_2_z) * im_01_i
    nz_10_r = (1.0 - 2.0 * noise_1_z) * im_10_r
    nz_10_i = (1.0 - 2.0 * noise_1_z) * im_10_i
    nz_11_r = (1.0 - 2.0 * noise_2_z) * (1.0 - 2.0 * noise_1_z) * im_11_r
    nz_11_i = (1.0 - 2.0 * noise_2_z) * (1.0 - 2.0 * noise_1_z) * im_11_i

    nx1_00_r = (1.0 - noise_2_x) * nz_00_r + noise_2_x * nz_01_r
    nx1_00_i = (1.0 - noise_2_x) * nz_00_i + noise_2_x * nz_01_i
    nx1_01_r = (1.0 - noise_2_x) * nz_01_r + noise_2_x * nz_00_r
    nx1_01_i = (1.0 - noise_2_x) * nz_01_i + noise_2_x * nz_00_i
    nx1_10_r = (1.0 - noise_2_x) * nz_10_r + noise_2_x * nz_11_r
    nx1_10_i = (1.0 - noise_2_x) * nz_10_i + noise_2_x * nz_11_i
    nx1_11_r = (1.0 - noise_2_x) * nz_11_r + noise_2_x * nz_10_r
    nx1_11_i = (1.0 - noise_2_x) * nz_11_i + noise_2_x * nz_10_i

    nx2_00_r = (1.0 - noise_1_x) * nx1_00_r + noise_1_x * nx1_10_r
    nx2_00_i = (1.0 - noise_1_x) * nx1_00_i + noise_1_x * nx1_10_i
    nx2_01_r = (1.0 - noise_1_x) * nx1_01_r + noise_1_x * nx1_11_r
    nx2_01_i = (1.0 - noise_1_x) * nx1_01_i + noise_1_x * nx1_11_i
    nx2_10_r = (1.0 - noise_1_x) * nx1_10_r + noise_1_x * nx1_00_r
    nx2_10_i = (1.0 - noise_1_x) * nx1_10_i + noise_1_x * nx1_00_i
    nx2_11_r = (1.0 - noise_1_x) * nx1_11_r + noise_1_x * nx1_01_r
    nx2_11_i = (1.0 - noise_1_x) * nx1_11_i + noise_1_x * nx1_01_i

    return nx2_00_r, nx2_00_i, nx2_01_r, nx2_01_i, nx2_10_r, nx2_10_i, nx2_11_r, nx2_11_i

@triton.jit
def crossing_kernel(sv_r_ptr, sv_i_ptr, noise_ptr, angles_ptr, width, nbatch, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(axis=1)
    elem_id = tl.program_id(axis=0)

    elem_mask = (1 << width) - 1
    elem = (elem_id & elem_mask) + ((elem_id & (0xffffffff ^ elem_mask)) << 3)

    block_offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_offsets < nbatch
    
    offsets_000 = nbatch * (elem + (0 << width)) + block_offsets
    offsets_100 = nbatch * (elem + (1 << width)) + block_offsets
    offsets_010 = nbatch * (elem + (2 << width)) + block_offsets
    offsets_110 = nbatch * (elem + (3 << width)) + block_offsets
    offsets_001 = nbatch * (elem + (4 << width)) + block_offsets
    offsets_101 = nbatch * (elem + (5 << width)) + block_offsets
    offsets_011 = nbatch * (elem + (6 << width)) + block_offsets
    offsets_111 = nbatch * (elem + (7 << width)) + block_offsets

    in_000_r = tl.load(sv_r_ptr + offsets_000, mask=mask)
    in_000_i = tl.load(sv_i_ptr + offsets_000, mask=mask)
    in_001_r = tl.load(sv_r_ptr + offsets_001, mask=mask)
    in_001_i = tl.load(sv_i_ptr + offsets_001, mask=mask)
    in_010_r = tl.load(sv_r_ptr + offsets_010, mask=mask)
    in_010_i = tl.load(sv_i_ptr + offsets_010, mask=mask)
    in_011_r = tl.load(sv_r_ptr + offsets_011, mask=mask)
    in_011_i = tl.load(sv_i_ptr + offsets_011, mask=mask)
    in_100_r = tl.load(sv_r_ptr + offsets_100, mask=mask)
    in_100_i = tl.load(sv_i_ptr + offsets_100, mask=mask)
    in_101_r = tl.load(sv_r_ptr + offsets_101, mask=mask)
    in_101_i = tl.load(sv_i_ptr + offsets_101, mask=mask)
    in_110_r = tl.load(sv_r_ptr + offsets_110, mask=mask)
    in_110_i = tl.load(sv_i_ptr + offsets_110, mask=mask)
    in_111_r = tl.load(sv_r_ptr + offsets_111, mask=mask)
    in_111_i = tl.load(sv_i_ptr + offsets_111, mask=mask)

    # Rz q[0]
    in_001_r, in_001_i = rz_subkernel(in_001_r, in_001_i, angles_ptr, 0)
    in_011_r, in_011_i = rz_subkernel(in_011_r, in_011_i, angles_ptr, 0)
    in_101_r, in_101_i = rz_subkernel(in_101_r, in_101_i, angles_ptr, 0)
    in_111_r, in_111_i = rz_subkernel(in_111_r, in_111_i, angles_ptr, 0)

    # Rz q[1]
    in_010_r, in_010_i = rz_subkernel(in_010_r, in_010_i, angles_ptr, 1)
    in_011_r, in_011_i = rz_subkernel(in_011_r, in_011_i, angles_ptr, 1)
    in_110_r, in_110_i = rz_subkernel(in_110_r, in_110_i, angles_ptr, 1)
    in_111_r, in_111_i = rz_subkernel(in_111_r, in_111_i, angles_ptr, 1)

    # Rz q[2]
    in_100_r, in_100_i = rz_subkernel(in_100_r, in_100_i, angles_ptr, 2)
    in_101_r, in_101_i = rz_subkernel(in_101_r, in_101_i, angles_ptr, 2)
    in_110_r, in_110_i = rz_subkernel(in_110_r, in_110_i, angles_ptr, 2)
    in_111_r, in_111_i = rz_subkernel(in_111_r, in_111_i, angles_ptr, 2)

    # U1q q[1]
    in_000_r, in_000_i, in_010_r, in_010_i = u1q_subkernel(in_000_r, in_000_i, in_010_r, in_010_i, noise_ptr + 0*nbatch, noise_ptr + 1*nbatch, block_offsets, mask, angles_ptr, 3, 4)
    in_001_r, in_001_i, in_011_r, in_011_i = u1q_subkernel(in_001_r, in_001_i, in_011_r, in_011_i, noise_ptr + 0*nbatch, noise_ptr + 1*nbatch, block_offsets, mask, angles_ptr, 3, 4)
    in_100_r, in_100_i, in_110_r, in_110_i = u1q_subkernel(in_100_r, in_100_i, in_110_r, in_110_i, noise_ptr + 0*nbatch, noise_ptr + 1*nbatch, block_offsets, mask, angles_ptr, 3, 4)
    in_101_r, in_101_i, in_111_r, in_111_i = u1q_subkernel(in_101_r, in_101_i, in_111_r, in_111_i, noise_ptr + 0*nbatch, noise_ptr + 1*nbatch, block_offsets, mask, angles_ptr, 3, 4)
    
    # Rz q[1]
    in_010_r, in_010_i = rz_subkernel(in_010_r, in_010_i, angles_ptr, 5)
    in_011_r, in_011_i = rz_subkernel(in_011_r, in_011_i, angles_ptr, 5)
    in_110_r, in_110_i = rz_subkernel(in_110_r, in_110_i, angles_ptr, 5)
    in_111_r, in_111_i = rz_subkernel(in_111_r, in_111_i, angles_ptr, 5)

    # RZZ q[1], q[2]
    in_000_r, in_000_i, in_010_r, in_010_i, in_100_r, in_100_i, in_110_r, in_110_i = rzz_subkernel(
        in_000_r, in_000_i, in_010_r, in_010_i, in_100_r, in_100_i, in_110_r, in_110_i,
        noise_ptr + 2 * nbatch, noise_ptr + 3 * nbatch, noise_ptr + 4 * nbatch, noise_ptr + 5 * nbatch, 
        block_offsets, mask, angles_ptr, 6
    )
    in_001_r, in_001_i, in_011_r, in_011_i, in_101_r, in_101_i, in_111_r, in_111_i = rzz_subkernel(
        in_001_r, in_001_i, in_011_r, in_011_i, in_101_r, in_101_i, in_111_r, in_111_i,
        noise_ptr + 2 * nbatch, noise_ptr + 3 * nbatch, noise_ptr + 4 * nbatch, noise_ptr + 5 * nbatch, 
        block_offsets, mask, angles_ptr, 6
    )

    # U1q q[1]
    in_000_r, in_000_i, in_010_r, in_010_i = u1q_subkernel(in_000_r, in_000_i, in_010_r, in_010_i, noise_ptr + 6*nbatch, noise_ptr + 7*nbatch, block_offsets, mask, angles_ptr, 7, 8)
    in_001_r, in_001_i, in_011_r, in_011_i = u1q_subkernel(in_001_r, in_001_i, in_011_r, in_011_i, noise_ptr + 6*nbatch, noise_ptr + 7*nbatch, block_offsets, mask, angles_ptr, 7, 8)
    in_100_r, in_100_i, in_110_r, in_110_i = u1q_subkernel(in_100_r, in_100_i, in_110_r, in_110_i, noise_ptr + 6*nbatch, noise_ptr + 7*nbatch, block_offsets, mask, angles_ptr, 7, 8)
    in_101_r, in_101_i, in_111_r, in_111_i = u1q_subkernel(in_101_r, in_101_i, in_111_r, in_111_i, noise_ptr + 6*nbatch, noise_ptr + 7*nbatch, block_offsets, mask, angles_ptr, 7, 8)

    # Rz q[2]
    in_100_r, in_100_i = rz_subkernel(in_100_r, in_100_i, angles_ptr, 9)
    in_101_r, in_101_i = rz_subkernel(in_101_r, in_101_i, angles_ptr, 9)
    in_110_r, in_110_i = rz_subkernel(in_110_r, in_110_i, angles_ptr, 9)
    in_111_r, in_111_i = rz_subkernel(in_111_r, in_111_i, angles_ptr, 9)

    # Rz q[1]
    in_010_r, in_010_i = rz_subkernel(in_010_r, in_010_i, angles_ptr, 10)
    in_011_r, in_011_i = rz_subkernel(in_011_r, in_011_i, angles_ptr, 10)
    in_110_r, in_110_i = rz_subkernel(in_110_r, in_110_i, angles_ptr, 10)
    in_111_r, in_111_i = rz_subkernel(in_111_r, in_111_i, angles_ptr, 10)

    # RZZ q[0], q[1]
    in_000_r, in_000_i, in_001_r, in_001_i, in_010_r, in_010_i, in_011_r, in_011_i = rzz_subkernel(
        in_000_r, in_000_i, in_001_r, in_001_i, in_010_r, in_010_i, in_011_r, in_011_i,
        noise_ptr + 8 * nbatch, noise_ptr + 9 * nbatch, noise_ptr + 10 * nbatch, noise_ptr + 11 * nbatch, 
        block_offsets, mask, angles_ptr, 11
    )
    in_100_r, in_100_i, in_101_r, in_101_i, in_110_r, in_110_i, in_111_r, in_111_i = rzz_subkernel(
        in_100_r, in_100_i, in_101_r, in_101_i, in_110_r, in_110_i, in_111_r, in_111_i,
        noise_ptr + 8 * nbatch, noise_ptr + 9 * nbatch, noise_ptr + 10 * nbatch, noise_ptr + 11 * nbatch, 
        block_offsets, mask, angles_ptr, 11
    )

    # Rz q[0]
    in_001_r, in_001_i = rz_subkernel(in_001_r, in_001_i, angles_ptr, 12)
    in_011_r, in_011_i = rz_subkernel(in_011_r, in_011_i, angles_ptr, 12)
    in_101_r, in_101_i = rz_subkernel(in_101_r, in_101_i, angles_ptr, 12)
    in_111_r, in_111_i = rz_subkernel(in_111_r, in_111_i, angles_ptr, 12)

    # U1q q[1]
    in_000_r, in_000_i, in_010_r, in_010_i = u1q_subkernel(in_000_r, in_000_i, in_010_r, in_010_i, noise_ptr + 12*nbatch, noise_ptr + 13*nbatch, block_offsets, mask, angles_ptr, 13, 14)
    in_001_r, in_001_i, in_011_r, in_011_i = u1q_subkernel(in_001_r, in_001_i, in_011_r, in_011_i, noise_ptr + 12*nbatch, noise_ptr + 13*nbatch, block_offsets, mask, angles_ptr, 13, 14)
    in_100_r, in_100_i, in_110_r, in_110_i = u1q_subkernel(in_100_r, in_100_i, in_110_r, in_110_i, noise_ptr + 12*nbatch, noise_ptr + 13*nbatch, block_offsets, mask, angles_ptr, 13, 14)
    in_101_r, in_101_i, in_111_r, in_111_i = u1q_subkernel(in_101_r, in_101_i, in_111_r, in_111_i, noise_ptr + 12*nbatch, noise_ptr + 13*nbatch, block_offsets, mask, angles_ptr, 13, 14)

    # Rz q[1]
    in_010_r, in_010_i = rz_subkernel(in_010_r, in_010_i, angles_ptr, 15)
    in_011_r, in_011_i = rz_subkernel(in_011_r, in_011_i, angles_ptr, 15)
    in_110_r, in_110_i = rz_subkernel(in_110_r, in_110_i, angles_ptr, 15)
    in_111_r, in_111_i = rz_subkernel(in_111_r, in_111_i, angles_ptr, 15)

    # RZZ q[1], q[2]
    in_000_r, in_000_i, in_010_r, in_010_i, in_100_r, in_100_i, in_110_r, in_110_i = rzz_subkernel(
        in_000_r, in_000_i, in_010_r, in_010_i, in_100_r, in_100_i, in_110_r, in_110_i,
        noise_ptr + 14 * nbatch, noise_ptr + 15 * nbatch, noise_ptr + 16 * nbatch, noise_ptr + 17 * nbatch, 
        block_offsets, mask, angles_ptr, 16
    )
    in_001_r, in_001_i, in_011_r, in_011_i, in_101_r, in_101_i, in_111_r, in_111_i = rzz_subkernel(
        in_001_r, in_001_i, in_011_r, in_011_i, in_101_r, in_101_i, in_111_r, in_111_i,
        noise_ptr + 14 * nbatch, noise_ptr + 15 * nbatch, noise_ptr + 16 * nbatch, noise_ptr + 17 * nbatch, 
        block_offsets, mask, angles_ptr, 16
    )

    # U1q q[1]
    in_000_r, in_000_i, in_010_r, in_010_i = u1q_subkernel(in_000_r, in_000_i, in_010_r, in_010_i, noise_ptr + 18*nbatch, noise_ptr + 19*nbatch, block_offsets, mask, angles_ptr, 17, 18)
    in_001_r, in_001_i, in_011_r, in_011_i = u1q_subkernel(in_001_r, in_001_i, in_011_r, in_011_i, noise_ptr + 18*nbatch, noise_ptr + 19*nbatch, block_offsets, mask, angles_ptr, 17, 18)
    in_100_r, in_100_i, in_110_r, in_110_i = u1q_subkernel(in_100_r, in_100_i, in_110_r, in_110_i, noise_ptr + 18*nbatch, noise_ptr + 19*nbatch, block_offsets, mask, angles_ptr, 17, 18)
    in_101_r, in_101_i, in_111_r, in_111_i = u1q_subkernel(in_101_r, in_101_i, in_111_r, in_111_i, noise_ptr + 18*nbatch, noise_ptr + 19*nbatch, block_offsets, mask, angles_ptr, 17, 18)

    # Rz q[2]
    in_100_r, in_100_i = rz_subkernel(in_100_r, in_100_i, angles_ptr, 19)
    in_101_r, in_101_i = rz_subkernel(in_101_r, in_101_i, angles_ptr, 19)
    in_110_r, in_110_i = rz_subkernel(in_110_r, in_110_i, angles_ptr, 19)
    in_111_r, in_111_i = rz_subkernel(in_111_r, in_111_i, angles_ptr, 19)

    # Rz q[1]
    in_010_r, in_010_i = rz_subkernel(in_010_r, in_010_i, angles_ptr, 20)
    in_011_r, in_011_i = rz_subkernel(in_011_r, in_011_i, angles_ptr, 20)
    in_110_r, in_110_i = rz_subkernel(in_110_r, in_110_i, angles_ptr, 20)
    in_111_r, in_111_i = rz_subkernel(in_111_r, in_111_i, angles_ptr, 20)

    tl.store(sv_r_ptr + offsets_000, in_000_r, mask=mask)
    tl.store(sv_i_ptr + offsets_000, in_000_i, mask=mask)
    tl.store(sv_r_ptr + offsets_001, in_001_r, mask=mask)
    tl.store(sv_i_ptr + offsets_001, in_001_i, mask=mask)
    tl.store(sv_r_ptr + offsets_010, in_010_r, mask=mask)
    tl.store(sv_i_ptr + offsets_010, in_010_i, mask=mask)
    tl.store(sv_r_ptr + offsets_011, in_011_r, mask=mask)
    tl.store(sv_i_ptr + offsets_011, in_011_i, mask=mask)
    tl.store(sv_r_ptr + offsets_100, in_100_r, mask=mask)
    tl.store(sv_i_ptr + offsets_100, in_100_i, mask=mask)
    tl.store(sv_r_ptr + offsets_101, in_101_r, mask=mask)
    tl.store(sv_i_ptr + offsets_101, in_101_i, mask=mask)
    tl.store(sv_r_ptr + offsets_110, in_110_r, mask=mask)
    tl.store(sv_i_ptr + offsets_110, in_110_i, mask=mask)
    tl.store(sv_r_ptr + offsets_111, in_111_r, mask=mask)
    tl.store(sv_i_ptr + offsets_111, in_111_i, mask=mask)

@triton.jit
def ctrl_cnot_kernel(sv_r_ptr, sv_i_ptr, ctrl_ptr, noise_ptr, angles_ptr, width_1, width_2, nbatch, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(axis=1)
    elem_id_1 = tl.program_id(axis=0)

    w2_mask = ((1 << width_2) - 1)
    elem_id_2 = (elem_id_1 & w2_mask) + ((elem_id_1 & (0xffffffff ^ w2_mask)) << 1)
    w1_mask = ((1 << width_1) - 1)
    elem = (elem_id_2 & w1_mask) + ((elem_id_2 & (0xffffffff ^ w1_mask)) << 1)

    block_offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_offsets < nbatch
    offsets_10 = nbatch * (elem + (1 << width_1) + (0 << width_2)) + block_offsets
    offsets_01 = nbatch * (elem + (0 << width_1) + (1 << width_2)) + block_offsets
    offsets_00 = nbatch * (elem + (0 << width_1) + (0 << width_2)) + block_offsets
    offsets_11 = nbatch * (elem + (1 << width_1) + (1 << width_2)) + block_offsets

    og_01_r = tl.load(sv_r_ptr + offsets_01, mask=mask)
    og_01_i = tl.load(sv_i_ptr + offsets_01, mask=mask)
    og_10_r = tl.load(sv_r_ptr + offsets_10, mask=mask)
    og_10_i = tl.load(sv_i_ptr + offsets_10, mask=mask)
    og_00_r = tl.load(sv_r_ptr + offsets_00, mask=mask)
    og_00_i = tl.load(sv_i_ptr + offsets_00, mask=mask)
    og_11_r = tl.load(sv_r_ptr + offsets_11, mask=mask)
    og_11_i = tl.load(sv_i_ptr + offsets_11, mask=mask)

    # U1q q[1]
    in_00_r, in_00_i, in_10_r, in_10_i = u1q_subkernel(og_00_r, og_00_i, og_10_r, og_10_i, noise_ptr + 0*nbatch, noise_ptr + 1*nbatch, block_offsets, mask, angles_ptr, 0, 1)
    in_01_r, in_01_i, in_11_r, in_11_i = u1q_subkernel(og_01_r, og_01_i, og_11_r, og_11_i, noise_ptr + 0*nbatch, noise_ptr + 1*nbatch, block_offsets, mask, angles_ptr, 0, 1)

    # RZZ q[0], q[1]
    in_00_r, in_00_i, in_01_r, in_01_i, in_10_r, in_10_i, in_11_r, in_11_i = rzz_subkernel(
        in_00_r, in_00_i, in_01_r, in_01_i, in_10_r, in_10_i, in_11_r, in_11_i,
        noise_ptr + 2 * nbatch, noise_ptr + 3 * nbatch, noise_ptr + 4 * nbatch, noise_ptr + 5 * nbatch, 
        block_offsets, mask, angles_ptr, 2
    )

    # Rz q[0]
    in_01_r, in_01_i = rz_subkernel(in_01_r, in_01_i, angles_ptr, 3)
    in_11_r, in_11_i = rz_subkernel(in_11_r, in_11_i, angles_ptr, 3)

    # U1q q[1]
    in_00_r, in_00_i, in_10_r, in_10_i = u1q_subkernel(in_00_r, in_00_i, in_10_r, in_10_i, noise_ptr + 6*nbatch, noise_ptr + 7*nbatch, block_offsets, mask, angles_ptr, 4, 5)
    in_01_r, in_01_i, in_11_r, in_11_i = u1q_subkernel(in_01_r, in_01_i, in_11_r, in_11_i, noise_ptr + 6*nbatch, noise_ptr + 7*nbatch, block_offsets, mask, angles_ptr, 4, 5)

    # Rz q[1]
    in_10_r, in_10_i = rz_subkernel(in_10_r, in_10_i, angles_ptr, 6)
    in_11_r, in_11_i = rz_subkernel(in_11_r, in_11_i, angles_ptr, 6)
    
    ctrl = tl.load(ctrl_ptr + block_offsets, mask=mask)
    out_00_r = (1.0 - ctrl) * og_00_r + ctrl * in_00_r
    out_00_i = (1.0 - ctrl) * og_00_i + ctrl * in_00_i
    out_01_r = (1.0 - ctrl) * og_01_r + ctrl * in_01_r
    out_01_i = (1.0 - ctrl) * og_01_i + ctrl * in_01_i
    out_10_r = (1.0 - ctrl) * og_10_r + ctrl * in_10_r
    out_10_i = (1.0 - ctrl) * og_10_i + ctrl * in_10_i
    out_11_r = (1.0 - ctrl) * og_11_r + ctrl * in_11_r
    out_11_i = (1.0 - ctrl) * og_11_i + ctrl * in_11_i

    tl.store(sv_r_ptr + offsets_01, out_01_r, mask=mask)
    tl.store(sv_i_ptr + offsets_01, out_01_i, mask=mask)
    tl.store(sv_r_ptr + offsets_10, out_10_r, mask=mask)
    tl.store(sv_i_ptr + offsets_10, out_10_i, mask=mask)
    tl.store(sv_r_ptr + offsets_00, out_00_r, mask=mask)
    tl.store(sv_i_ptr + offsets_00, out_00_i, mask=mask)
    tl.store(sv_r_ptr + offsets_11, out_11_r, mask=mask)
    tl.store(sv_i_ptr + offsets_11, out_11_i, mask=mask)

@dataclasses.dataclass
class Statevector:
    r: torch.Tensor
    i: torch.Tensor
    nqubits: int

    def __init__(self, nqubits: int, batch_size: int, device: torch.device = 'cuda'):
        self.r = torch.zeros((2**nqubits, batch_size), device=device)
        self.i = torch.zeros((2**nqubits, batch_size), device=device)
        self.r[0, :] = 1.0
        self.nqubits = nqubits
        self.cnot_angles = torch.tensor([-torch.pi/2, torch.pi/2, torch.pi/2, -torch.pi/2, torch.pi/2, torch.pi, -torch.pi/2], device=device)

    def pauli(self, ctrl_z: torch.Tensor, ctrl_x: torch.Tensor, qubit: int):
        ctrl_z = ctrl_z.contiguous()
        ctrl_x = ctrl_x.contiguous()
        BLOCK_SIZE = 128
        grid = (self.r.shape[0] // 2, triton.cdiv(self.r.shape[1], BLOCK_SIZE))
        ctrl_pauli_kernel[grid](self.r, self.i, ctrl_z, ctrl_x, qubit, self.r.shape[1], BLOCK_SIZE=BLOCK_SIZE)

    def rz(self, ctrl: torch.Tensor, qubit: int, theta: float):
        ctrl = ctrl.contiguous()
        BLOCK_SIZE = 128
        grid = (self.r.shape[0] // 2, triton.cdiv(self.r.shape[1], BLOCK_SIZE))
        ctrl_rz_kernel[grid](self.r, self.i, ctrl, qubit, self.r.shape[1], theta, BLOCK_SIZE=BLOCK_SIZE)

    def var_rz(self, theta: torch.Tensor, qubit: int):
        theta = theta.contiguous()
        BLOCK_SIZE = 128
        grid = (self.r.shape[0] // 2, triton.cdiv(self.r.shape[1], BLOCK_SIZE))
        var_rz_kernel[grid](self.r, self.i, theta, qubit, self.r.shape[1], BLOCK_SIZE=BLOCK_SIZE)

    def u1q(self, ctrl: torch.Tensor, noise_x: torch.Tensor, noise_z: torch.Tensor, qubit: int, theta: float, phi: float):
        ctrl = ctrl.contiguous()
        noise_x = noise_x.contiguous()
        noise_z = noise_z.contiguous()
        BLOCK_SIZE = 128
        grid = (self.r.shape[0] // 2, triton.cdiv(self.r.shape[1], BLOCK_SIZE))
        ctrl_u1q_kernel[grid](self.r, self.i, ctrl, noise_x, noise_z, qubit, self.r.shape[1], theta, phi, BLOCK_SIZE=BLOCK_SIZE)

    def cnot(self, ctrl: torch.Tensor, noise: torch.Tensor, q1: int, q2: int):
        ctrl = ctrl.contiguous()
        noise = noise.contiguous()
        BLOCK_SIZE = 128
        grid = (self.r.shape[0] // 4, triton.cdiv(self.r.shape[1], BLOCK_SIZE))
        if q2 > q1:
            q1, q2 = q2, q1
        ctrl_cnot_kernel[grid](self.r, self.i, ctrl, noise, self.cnot_angles, q1, q2, self.r.shape[1], BLOCK_SIZE=BLOCK_SIZE)

    def crossing(self, noise: torch.Tensor, angles: torch.Tensor, qlow: int):
        noise = noise.contiguous()
        angles = angles.contiguous()
        BLOCK_SIZE = 128
        grid = (self.r.shape[0] // 8, triton.cdiv(self.r.shape[1], BLOCK_SIZE))
        crossing_kernel[grid](self.r, self.i, noise, angles, qlow, self.r.shape[1], BLOCK_SIZE=BLOCK_SIZE)

    def measure(self):
        probs = self.r * self.r + self.i * self.i
        dist = torch.distributions.Categorical(probs=probs.T)
        indices = dist.sample()
        return torch.stack([
            (indices & (1 << i)) != 0
            for i in range(self.nqubits)
        ], dim=-1)

def torch_compute(spec: dict, ops: list, nqubits: int, noise: torch.Tensor, cbits: torch.Tensor, crossing_angles: torch.Tensor, mnoise: torch.Tensor | None):
    sv = Statevector(nqubits, cbits.shape[1], device=noise.device)
    all_zeros = torch.zeros(cbits.shape[1], device=noise.device)
    all_ones = torch.ones(cbits.shape[1], device=noise.device)

    if spec["spam"]["prepare"] != 0.0:
        for i in range(nqubits):
            sv.pauli(all_zeros, noise[i - nqubits, :], i)

    cidx = 0
    for name, params, qubits, cbit, cond, nidx in ops:
        if cbit is not None:
            ctrl = cbits[cbit, :] if cond else 1.0 - cbits[cbit, :]
        else:
            ctrl = all_ones

        if name == 'Rz':
            sv.rz(ctrl, qubits[0], params[0])
        elif name == 'U1q':
            sv.u1q(ctrl, noise[nidx, :], noise[nidx + 1, :], qubits[0], params[0], params[1])
        elif name == 'cnot':
            sv.cnot(ctrl, noise[nidx:nidx+8, :], qubits[0], qubits[1])
        elif name == 'crossing':
            if mnoise is not None:
                sv.var_rz(mnoise[cidx, 0, :], qubits[0])
                sv.var_rz(mnoise[cidx, 1, :], qubits[1])
                sv.var_rz(mnoise[cidx, 2, :], qubits[2])
                cidx += 1
            sv.crossing(noise[nidx:nidx+20, :], crossing_angles[params[0]], qubits[0])
        else:
            raise ValueError(f"Unsupported gate: {name}")

    return sv.measure()

def sample_noise(nprobs, samples, device):
    s = []
    for n, k in nprobs:
        if k == 1:
            s.append((torch.rand(size=(samples,), device=device) < n).to(float))
        elif k == 2:
            which = (torch.rand(size=(samples,), device=device) < n) * torch.randint(1, 4, size=(samples,), device=device)
            s.append(((which & 1) != 0).to(float))
            s.append(((which & 2) != 0).to(float))
        elif k == 4:
            which = (torch.rand(size=(samples,), device=device) < n) * torch.randint(1, 16, size=(samples,), device=device)
            s.append(((which & 1) != 0).to(float))
            s.append(((which & 2) != 0).to(float))
            s.append(((which & 4) != 0).to(float))
            s.append(((which & 8) != 0).to(float))
    return torch.stack(s, dim=0)

def sample_memory_error(mprobs, samples, device, memory_scale: float):
    s = []
    for m in mprobs:
        s.append(torch.stack([
            torch.fmod(torch.rand(size=(samples,), device=device) * memory_scale * m[0], 2*np.pi),
            torch.fmod(torch.rand(size=(samples,), device=device) * memory_scale * m[1], 2*np.pi),
            torch.fmod(torch.rand(size=(samples,), device=device) * memory_scale * m[2], 2*np.pi),
        ], dim=0))
    return torch.stack(s, dim=0)
    
#Add measurement errors to a set of noiseless samples
def inject_measurement_errors(spec: dict, noiseless_bits: torch.Tensor, scale: float, device=torch.device) -> torch.Tensor:
    # The probability of a measurement error depends on the noiseless value
    probs = torch.where(noiseless_bits, spec["spam"]["measure"]["one"] * scale, spec["spam"]["measure"]["zero"] * scale)
    # Flip samples at random with these probabilities
    flips = torch.rand(size=probs.shape, device=device) < probs
    noisy_bits = noiseless_bits ^ flips
    return noisy_bits

crossing_angles = [
    [
        6.113998825707325, 0.5774678903872845, 0.002366522379226622, 1.269374006243357, 1.1397682926441883, 0.977081323027324, 1.2283022625961766, 1.269374006243357, 
        0.20355922467789583, 6.164536741772381, 0.38364469373232135, 1.5707963267948966, 6.138212523292868, 1.2134473565378714, 0.5872039184102174, 0.7512028663148562, 
        1.2873813780687242, 1.2134473565378716, 5.767380816383591, 6.085308084848585, 0.13562709697596145
    ], [
        2.092082062086321, 1.0037863046985274, 0.4281846018192521, 2.045869596521337, 6.115248944737707, 1.4762802414286598, 1.084461799655791, 2.045869596521337, 
        5.534398332232365, 0.40909053314684196, 6.214051297572579, 1.256637061435917, 1.6778291222214303, 1.9568778945948029, 1.6953531383176037, 0.9686223456212558, 
        1.0133647107157195, 1.9568778945948029, 0.5357475410647861, 0.4193619264698243, 0.8058039146666414
    ], [
        0.004259073689200102, 2.4144043267079525, 5.483637772818747, 1.3030777948351726, 6.223182009350231, 1.7182613248415903, 4.860277925257167, 1.8385148587546147, 
        0.2353506450898184, 5.765951510915193, 2.1207582073051037, 0.55176529073289, 0.07229416629587705, 1.8755694453642386, 0.3180467802708876, 1.6310618471227207, 
        4.724057259848325, 1.2660232082255543, 0.38998058006234054, 5.638457805665861, 1.610394750411136
    ], [
        0.022336783355811452, 5.018923200885862, 0.7205344807083275, 5.059476716857704, 0.9549142718292872, 5.714613480919112, 1.622091675755497, 1.223708590461057, 
        2.0084341213411645, 0.6884025934291275, 0.8341141529256176, 1.0867763566165753, 6.090987820727624, 1.2202382125214735, 0.6681917650446233, 0.9009458881075508, 
        1.5604019312774595, 1.2202382124322337, 6.2711322381265004, 0.5955587313633421, 0.7544111929015721
    ], [
        0.9883339646672431, 2.268851422299875, 0.5267612022064855, 2.3988657393531896, 0.15527975968553745, 0.8409496864788594, 1.1526594630683944, 0.7427269142365814, 
        2.1488889092327783, 0.6800848692899353, 2.3197341866388057, 1.5707963267959915, 0.5824623621256303, 0.9895043862011541, 4.468623095875707, 1.7232901408334407, 
        0.8323165352915101, 2.152088267388597, 0.7410444648210495, 0.36395025530099406, 1.8577728529477715
    ], [
        0.3231058751695687, 1.0143793062492896, 0.7563100879633913, 5.060461875308376, 1.9147744979655772, 1.7903648519435162, 1.5270021516372532, 1.9188692217185834, 
        2.090548847956553, 1.2570683191633465, 2.512970817383275, 1.0867763573769063, 6.1299401360690275, 1.44389753540808, 0.4938870729140574, 1.9472529166829555, 
        1.2429656666123259, 4.5854901889978725, 0.5425130026195453, 0.33003501168608773, 3.0410170160275865
    ], [
        6.029664391382892, 0.2894909759765098, 0.6770624303794229, 5.007622292078912, 2.5123519075728846, 6.06030132395869, 1.4931290237087105, 5.007622292027325, 
        0.6410042922059723, 0.7319807903353396, 0.38564412639844264, 0.551765290704361, 0.17696767588522158, 1.2778456263714364, 2.1301790023939655, 0.13769826566995855, 
        1.484704508245479, 5.005339680900787, 3.7525817754266773, 0.5524656113086445, 0.27868150738772074
    ], [
        4.368510795846241, 3.521600670761039, 3.787785537362332, 4.278991151293781, 2.894307318917594, 5.736565919527429, 4.189220482575109, 1.1373984973652114, 
        3.395315760164264, 3.4321290654064214, 2.5196202357536635, 1.2566370619635308, 4.427948630006334, 4.0705379433012965, 5.286617474716735, 0.5727178148517069, 
        4.905166289922988, 5.354240017423926, 4.481316272857535, 4.089818956507621, 4.944486864904796
    ], [
        1.784398336002326, 3.7343287794727047, 1.5584202183484237, 5.068894975527158, 4.745907473387575, 1.812063189777085, 4.42789691769643, 1.2142903315795692, 
        4.7026822737322505, 1.161274166706157, 6.174951401404165, 1.5707963251184012, 1.6713535873484062, 5.014859200291873, 1.4528557250393916, 0.8445925954372839, 
        4.370811665423652, 1.2683261071536265, 0.3850746787495538, 0.7360575248852044, 1.5689818110188298
    ]
]

eigenphases = [
    3.141592653589793, 6.283185307179585, 3.1415926535898038,
    3.8705521639528957, 3.1415926535882726, 6.2831853071795845,
    3.1415926532457696, 6.283185302741501, 3.1415926634545244,
]

if __name__ == "__main__":
    device = torch.device(f'cuda:{args.gpu}')
    spec = json.load(open(args.SPEC, "r"))
    shots = args.SHOTS
    if shots != 1:
        raise ValueError("Only one shot per bitstring is supported")
    
    if args.bits is None:
        raise ValueError("Only simulations with bitstrings are supported")

    if args.output is None:
        outputs = [None]*len(args.BRAID)
    else:
        outputs = args.output

    if args.time_limit is not None:
        target_time = time.time() + args.time_limit
    else:
        target_time = float('inf')

    # Read a circuit from disk
    for braidfile, bitfile, outputfile in zip(args.BRAID, args.bits, outputs):
        print(braidfile, bitfile)
        bspec = json.load(open(braidfile, "r"))
        if not args.conjugate:
            braid = Braid.from_word(bspec["optimized"]["word"])
        else:
            braid = Braid.from_word([-c for c in bspec["optimized"]["word"]])
        cbits = torch.tensor(np.load(bitfile)).to(device).to(float).T.contiguous()
        ncbits = cbits.shape[1]
        
        out_file = pathlib.Path(outputfile) if outputfile is not None else pathlib.Path(braidfile).with_suffix(".out.npy")

        if out_file.exists():
            print(f"`{out_file}` already exists, skipping")
            continue

        # Get the operations for the circuit
        ops, nqubits, nprobs, mprobs = construct_cfev(spec, braid, args.scale)
        crossing_angles_d = torch.tensor(crossing_angles, device=device)
        if args.zero_angles:
            crossing_angles_d.zero_()

        results = []
        for idx in tqdm.trange(0, cbits.shape[1], args.batchsize):
            if time.time() >= target_time:
                break

            cbits_batch = cbits[:, idx:idx+args.batchsize]
            noise = sample_noise(nprobs, cbits_batch.shape[1], device)
            if args.memory_error is not None:
                mnoise = sample_memory_error(mprobs, cbits_batch.shape[1], device, args.memory_error)
            else:
                mnoise = None
            measurements = torch_compute(spec, ops, nqubits, noise, cbits_batch, crossing_angles_d, mnoise)
            noisy_measurements = inject_measurement_errors(spec, measurements, args.scale, device)
            results.append(noisy_measurements)

        if time.time() >= target_time:
            break

        result = torch.cat(results, dim=0).cpu().numpy()
        np.save(out_file, result[:, None, :])
        print(f"wrote shot values of shape {result[:, None, :].shape} to `{out_file}`")