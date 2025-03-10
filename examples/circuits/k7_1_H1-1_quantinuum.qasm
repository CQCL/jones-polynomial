OPENQASM 2.0;

opaque U1q(p0, p1) q0;
opaque Rz(p0) q0;
opaque RZZ(p0) q0, q1;

qreg q[3];
creg c[2];

U1q(1.5707963, -1.5707963) q[1];
Rz(3.14159265) q[1];
if (c[0] == 1) Rz(-1.5707963267948966) q[1];
if (c[1] == 1) U1q(-1.5707963, 1.5707963) q[2];
if (c[1] == 1) RZZ(1.5707963) q[1], q[2];
if (c[1] == 1) Rz(-1.5707963) q[1];
if (c[1] == 1) U1q(1.5707963, 3.14159265) q[2];
if (c[1] == 1) Rz(-1.5707963) q[2];
Rz(0.2526301847421385) q[0];
Rz(-0.06015253055349863) q[1];
Rz(-0.3297139281085655) q[2];
U1q(-1.358812625006971, 0.28788372906880827) q[1];
Rz(0.7191536327515209) q[1];
RZZ(-1.3362159285440958) q[1], q[2];
U1q(-1.3632917693285922, 2.8027523953406104) q[1];
Rz(-0.25399280476139047) q[2];
Rz(0.46243370504203835) q[1];
RZZ(-0.5512873957682142) q[0], q[1];
Rz(-0.18186458916001513) q[0];
U1q(-1.3666051588373687, -2.112952474462056) q[1];
Rz(-0.28778509853919637) q[1];
RZZ(-1.3335845346427686) q[1], q[2];
U1q(1.362065831946614, 2.5393537644806865) q[1];
Rz(-0.24445483529081696) q[2];
Rz(0.26856202030735066) q[1];
if (c[1] == 1) U1q(-1.5707963, 1.5707963) q[2];
if (c[1] == 1) RZZ(1.5707963) q[1], q[2];
if (c[1] == 1) Rz(-1.5707963) q[1];
if (c[1] == 1) U1q(1.5707963, 3.14159265) q[2];
if (c[1] == 1) Rz(-1.5707963) q[2];
Rz(-0.5905545131063051) q[1];
U1q(1.5707963, -1.5707963) q[1];
Rz(3.14159265) q[1];
