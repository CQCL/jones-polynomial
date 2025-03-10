## Less Quantum, More Advantage: An End-to-End Quantum Algorithm for the Jones Polynomial

This repository contains an implemetation of the end-to-end pipeline described in the paper ["Less Quantum, More Advantage: An End-to-End Quantum Algorithm for the Jones Polynomial"](https://arxiv.org/abs/2503.05625) as well the code and data necessary to reproduce the results of the paper. In particular, the pipeline allows you to:
* Generate and minimize random braids, 
* Generate benchmark sets of braids with known Jones polynomials,
* Compile quantum circuits for any device to evaluate the Jones polynomial
* Simulate these circuits for small braids with a noise model
* Perform the classical pre- and post-processing for the quantum algorithm

For example of how to simulate the whole algorithm end-to-end on a small instance, we can do as follows.

1. Create a folder to hold the output: 
```bash
    mkdir test/ 
```
2. Generate a random benchmark braid on ten strands: 
```bash
    python generate.py -o test/ benchmark -i 10 30
```
3. Compile the quantum part of the algorithm for the Quantinuum H1-1 device:
```bash
    python compile.py -o test/ specs/H1-1.json ansatze/quantinuum.qasm test/b_n10_l30.json
```
4. Sample bitstrings for 10000 shots:
```bash
    python preprocessing.py -o test/ 10000 test/b_n10_l30.json
```
5. Simulate the circuit to obtain measurement results:
```bash
    python simulate.py specs/H1-1.json 1 test/b_n10_l30_H1-1_quantinuum.qasm -o test/b_n10_l30.out.npy -b test/b_n10_l30.npy 
```
6. Produce an estimate of the Jones polynomial:
```bash
    python postprocessing.py test/b_n10_l30.npy test/b_n10_l30.out.npy test/b_n10_l30.json -b -r10000 -o test/
```
7. Now you can compare the estimated value of the Jones polynomial in `test/b_n10_l30.out.res.json` with the true value in `test/b_n10_l30.json`! The phase should be roughly accurate but there will be a fairly large discrepancy in the amplitude due to the simulated gate noise.

#### Installation

The pipeline is given as a set of standalone Python scripts which do not need to be installed. See `requirements.txt` for a list of dependencies. The [`tl-tensor` package](https://github.com/tlaakkonen/tl-tensor) is additionally required **only** for estimating the cost of the `bracket` classical baseline.

### Generation

You can generate random brickwall braids with
```bash
    python generate.py -o {output folder} random {strands} {brickwall layers} -c {number of braids}
```
You can generate benchmark sets of braids with a known Jones polynomial using
```bash
    python generate.py -o {output folder} benchmark {strands} {brickwall layers} -i -c {number of braids}
```
In either case, for each braid, a JSON file will be emitted containing information about it, such as the braid word, Jones polynomial (if known), and statistics. You can also generate this file for a given knot or link, specified by a braid word - e.g for the 4_1 knot
```bash
    python generate.py -o {output folder} create -w "-1 2 -1 2"
```
or a planar diagram code of a link (a code in the form 'a b c d e f g h ...' will be interpreted as the crossings (a, b, c, d), (e, f, g, h), etc) - e.g for the trefoil
```bash
    python generate.py -o {output folder} create -p "1 4 2 5 3 6 4 1 5 2 6 3"
```
or a Dowker-Thistlethwaite code - e.g for the 5_2 knot:
```bash
    python generate.py -o {output folder} create -d "4 8 10 2 6"
```
A variety of options for braid word minimization are available, see the help text of `generate.py` for more information.

### Compilation

To compile a braid, you will need a device specification file, as well as braid generators compiled for this device. We include specification files and precompiled generators for the Quantinuum H1-1 device, which can also be used on all H-series machines. See the section on finetuning for how to obtain these for other devices.

Given a set of JSON files representing braids, a specification file, and the compiled braid generators, you can use the following command to produce a QASM file for each braid
```bash
    python compile.py -o {output folder} specs/{machine}.json ansatze/{machine}.qasm {braid}.json ... {braid}.json
```
It is expected that the compiled braid generators live in the `circuits/` folder, and have the file names as assigned by `generate.py` - if you have them in a different folder, you can configure this with the `-c` option. 

The QASM files produced contain the quantum part of the algorithm. They include a classical input register `c` which must be filled in with a different randomly generate bitstring for each run, see the 'Classical Pre- and Post-processing' section.

### Classical Pre- and Post-processing

The quantum part of the algorithm requires running a circuit that is parameterized by a classical bitstring many times with many different bitstrings, and collecting one measurement for each bitstring. The bitstrings are drawn from a specific distribution and can be generated using
```bash
    python preprocessing.py {shots} {braid}.json
```
and are output as a 2D integer NumPy array (in `.npy` format). Where `{braid}.json` refers to the JSON file produced by `generate.py` for a given braid.

Given a JSON file for a braid, the 2d NumPy array of sampled bitstrings, and a 2D NumPy array of measurement results, we can calculate an estimate of the Jones polynomial evaluated at the fifth root of unity (we assume that each row in the results array corresponds to the same row of the bitstring array). This can be done like
```bash
    python postprocess.py {bitstrings}.npy {measurement results}.npy {braid}.json
```
a JSON file will be emitted with an estimate of the Jones polynomial. Here `{bitstrings}.npy` refers to the preprocessing file created by `preprocessing.py`, `{measurement results}.npy` refers to a `{shots} x 1 x {qubits}` boolean NumPy array containing the measurement outcomes (for instance, as produced by `simulate.py`) and `{braid}.json` refers to the JSON file created by `generate.py`. To estimate confidence intervals for the resulting values using bootstrapping you can use the `-b -r {bootstrap samples}` options.

### Execution

If you have access to a quantum device and want to run this algorithm, the full pipeline is as follows:

1. Generate a braid JSON file using `generate.py`
2. Sample the random classical bitstrings using `preprocessing.py`
3. Compile the quantum part of the algorithm using `compile.py`
4. Execute the compiled quantum circuit once for each sampled bitstring, recording the measurement results.
5. Using `postprocessing.py`, estimate the Jones polynomial using the braid JSON file, output of `preprocessing.py` and the measurement results.

### Simulation

If you don't have access to an actual quantum device, a noisy statevector simulator implementation based on `qiskit-aer` is included. Given a QASM file produced above, as well as sampled bitstrings, and a device specification file, this can produce simulated measurement results with the noise model given in the specification. For example, it can be used like
```bash
    python simulate.py specs/{machine}.json 1 {compiled circuit}.qasm -b {bitstrings}.npy
```
Where `{compiled circuit}` refers to the circuit generated by `compile.py` and `{bitstrings}` refers to the file produced by `preprocessing.py` discussed in the next section. Note that if a suitable version of `qiskit-aer-gpu` is installed, GPU acceleration is available.

### Finetuning

To compile the braid generators for a given device, you need to provide a device specification file, along with an ansatz specification. We use this to variationally compile the braid generators for a given machine in a hardware-efficient manner.

The specification contains details about the native gateset of the device, along with an error model for the machine. We provide such files for all the Quantinuum machines, as well as the hypothetical machines with $5\cdot 10^{-4}$ and $1 \cdot 10^{-4}$ error. See these for an example of how to construct your own, as well as the `specs/README.md` file.

The ansatz specification gives a template circuit to optimize to produce the correct operation using the native gateset of the device. Not all ansatzes are capable of representing the given unitary, so you may need to experiment to find one which is hardware-efficient for your target device and also achieves good loss values. An example using only three 2-qubit gates is given that is valid for Quantinuum's H-series devices. All parameters are assumed to be in the range $[0, 2\pi]$. Given these two files, you can produce the braid generators using
```bash
    python finetune.py -p specs/{machine}.json ansatze/{machine}.qasm circuits/
```
If it fails to converge, you can adjust the learning rate and number of optimization steps with the `-l` and `-s` flags.

## Classical Baselines and Benchmarking Data

See `data/README.md` for information about the classical algorithms for computing the Jones polynomial and the noisy circuit simulation benchmarking data.

## Cite

If you find this useful in your research, consider citing our paper:
```
@misc{2503.05625,
      title={Less Quantum, More Advantage: An End-to-End Quantum Algorithm for the Jones Polynomial}, 
      author={Tuomas Laakkonen and Enrico Rinaldi and Chris N. Self and Eli Chertkov and Matthew DeCross and David Hayes and Brian Neyenhuis and Marcello Benedetti and Konstantinos Meichanetzidis},
      year={2025},
      eprint={2503.05625},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2503.05625}, 
}
```

