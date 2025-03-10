OPENQASM 2.0;

opaque U1q(theta, phi) q;
opaque Rz(lam) q;
opaque RZZ(theta) a, b;

qreg q[3];

Rz(0) q[0];
Rz(0) q[1];
U1q(0, 0) q[1];
Rz(0) q[1];
Rz(0) q[2];
RZZ(0) q[1], q[2];
U1q(0, 0) q[1];
Rz(0) q[1];
Rz(0) q[2];
RZZ(0) q[0], q[1];
Rz(0) q[0];
U1q(0, 0) q[1];
Rz(0) q[1];
RZZ(0) q[1], q[2];
U1q(0, 0) q[1];
Rz(0) q[1];
Rz(0) q[2];
