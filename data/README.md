This folder contains the code to implement the classical baseline algorithms considered in paper, as well as the data necessary to reproduce all the plots. The code to process the raw data is given in `collect_data.ipynb`, and the plots can be produced using `plots.ipynb`.

### Classical Baselines

The benchmark dataset used for the classical baselines is given in `raw_data/braids_n10_29_l5_50_c50_d8_1024.pkl`.

#### MPO Approximation (`mpo-proj`)

`mpo-proj` is implemented in `code/approx_tn.py` and `code/tn_proj_mpo.py`. See the help text for `code/tn_proj_mpo.py` for more details. It was run as described in the paper and the compressed raw data can be found in `raw_data/mpo_data_d0.pkl.gz`, and the processed results are given in `processed/mpo_aggregate.npz` and `processed/mpo_props.npz`. The raw data needs to be uncompressed before running the plotting code.

#### Exact Tensor Network Contraction (`tn-proj`)

`tn-proj` is implemented in `exact_tensor_costs.py`. The processed data is given in `processed/exact_contractions.npz`.

#### Shor-Jordan Statevector Simulation (`sv-shor`)

The time-to-solution calculations for this method are given entirely in `plots.ipynb`.

#### Kauffman Bracket Tensor Network (`bracket`)

This method is implemented based on the code in the [`tl-tensor` repository](https://github.com/tlaakkonen/tl-tensor), which implements a similar algorithm for computing the Jones polynomial from the planar diagram of a knot or link up to medium scales (e.g ~100 crossings). The underlying tensor network structure is the same as in Kauffman's method so this is to estimate the cost of both methods. The `tl-tensor` package must be installed from the link above before this estimation can be run. The cost estimation code is given in `code/tl_tensor_estimate.py`. The processed data is given in `processed/kbtensor_costs.npz`. 

### Noisy Circuit Simulations

As outlined in the paper, noisy statevector circuit simulations were used to characterize the effect of gate noise on the accuracy of the final estimation of the Jones polynomial. This can be done using the `simulate.py` script in the root folder, but this is very inefficient (as `qiskit-aer` is implemented in a way that prioritizes many shots of a single circuit, rather than a single shot of many circuits). Hence, a simulator based on the [Triton language](https://triton-lang.org/main/index.html) is implemented in `simulate_triton.py`, the command line arguments are the same as for `simulate.py`. The raw data for the 5e-4 and 1e-4 two-qubit error levels are given in `raw_data/results_small_cross.npz` and `raw_data/results_small_1e-4_cross.npz` (this also includes details of the braids in the benchmark set). Processed data is given in `processed/sims_1_props.npz` and `processed/sims_5_props.npz`.

### Evaluation on the H2-2 Quantum Computer

The relevant files are given in the `device_run` folder, and includes data from the H2-2 quantum computer and the H1-1E emulator. The braid used in the demonstration is given in `device_run/b_n15_l15.json`, and its conjugate is given in `device_run/b_n15_l15_conj.json`. 

For the H1-1E emulator, only non-Fibonacci error detection and the conjugate trick were used. The corresponding compiled circuits for the four parts (real and imaginary parts of the braid and its conjugate) are given in `device_run/b_n15_l15_{conj}_H2_{real/imag}.qasm`. The measurement results are given in `device_run/results/h1-1e`. The results were processed in `collect_data.ipynb` and the data is in `processed/emulator.npz`. 

For the H2-2 run, the shot-level conjugate trick was used. The combined circuit with the four parts interleaved is given in `device_run/b_n15_l15_bi_br_ci_cr.qasm`. The results are given in `results/h2-2/`, code to process them with bootstrapping and apply the shot-level conjugate trick is given in `interleaved_replicas.py`. The resulting data is given in `processed/h2-2_interleaved.pkl`. 

In both cases, to switch the classical input bitstring for each shot, all the qubits are initialized in the $\ket{+}$ state and measured in the computational basis to generate entropy. Then using the [WASM-based classical compute utilities of the H-series devices](https://docs.quantinuum.com/systems/trainings/getting_started/qec_decoder_toolkit/qec_decoder_toolkit.html#webassembly-in-the-quantinuum-stack), a bitstring is sampled on the fly and used to parameterize the quantum circuit. The WASM and corresponding Rust source code for this procedure is given in the `wasm` folder. A recent Rust compiler is required to build it.
