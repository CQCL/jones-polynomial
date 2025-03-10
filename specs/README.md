We provide specification files for four noise profiles:
1. `H1-1.json` this is intended to mimic Quantinuum's H1-1 device and has error rates taken from the datasheet
2. `H-scale-1e-4`, `H-scale-5e-4`. These two mimic the values of H1-1 but scaled down so that the two-qubit error rates are 1e-4 and 5e-4 respectively.
4. `H-flat-1e-4`, `H-flat-5e-4`. These are the same but the error of the variable-angle two-qubit gate does not depend on the angle.
