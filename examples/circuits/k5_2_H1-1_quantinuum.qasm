OPENQASM 2.0;

opaque U1q(p0, p1) q0;
opaque Rz(p0) q0;
opaque RZZ(p0) q0, q1;

qreg q[4];
creg c[3];

U1q(1.5707963, -1.5707963) q[1];
Rz(3.14159265) q[1];
if (c[0] == 1) Rz(-1.5707963267948966) q[1];
if (c[1] == 1) U1q(-1.5707963, 1.5707963) q[2];
if (c[1] == 1) RZZ(1.5707963) q[1], q[2];
if (c[1] == 1) Rz(-1.5707963) q[1];
if (c[1] == 1) U1q(1.5707963, 3.14159265) q[2];
if (c[1] == 1) Rz(-1.5707963) q[2];
if (c[2] == 1) U1q(-1.5707963, 1.5707963) q[3];
if (c[2] == 1) RZZ(1.5707963) q[1], q[3];
if (c[2] == 1) Rz(-1.5707963) q[1];
if (c[2] == 1) U1q(1.5707963, 3.14159265) q[3];
if (c[2] == 1) Rz(-1.5707963) q[3];
Rz(1.6788048119099352) q[0];
Rz(1.389184869822185) q[1];
Rz(0.9388801590056095) q[2];
U1q(1.4929690501868802, -0.02146858359059794) q[1];
Rz(1.3879946366092946) q[1];
RZZ(1.1208015329361796) q[1], q[2];
U1q(1.493927274077808, -0.6561788535491784) q[1];
Rz(0.6944379926432168) q[2];
Rz(1.4033426127718334) q[1];
RZZ(1.5670097865191353) q[0], q[1];
Rz(1.7739379706067475) q[0];
U1q(-1.492562923475675, -1.1330286067647313) q[1];
Rz(0.8294342552805627) q[1];
RZZ(1.121033391933071) q[1], q[2];
U1q(1.4917484960326726, 0.8154828926955129) q[1];
Rz(0.5617571120766232) q[2];
Rz(0.5980743505425773) q[1];
Rz(-0.16595439813837065) q[1];
Rz(0.6982003258878722) q[2];
Rz(-0.3339580253965829) q[3];
U1q(1.4290871319977352, 1.6631465927580007) q[2];
Rz(0.28255045220541325) q[2];
RZZ(1.1370519044424658) q[2], q[3];
U1q(1.4310758013167295, -0.05870573718126487) q[2];
Rz(-0.13294532414044183) q[3];
Rz(-0.07535171856420175) q[2];
RZZ(1.5684865236573335) q[1], q[2];
Rz(-0.1511905917952387) q[1];
U1q(1.4304679870112307, 0.8509980185141269) q[2];
Rz(0.6189273513735188) q[2];
RZZ(1.1372381174921393) q[2], q[3];
U1q(1.4284837092111855, -0.5342943023551773) q[2];
Rz(-0.8305295866908198) q[3];
Rz(-0.42932991480443716) q[2];
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
Rz(-0.16595439813837065) q[1];
Rz(0.6982003258878722) q[2];
Rz(-0.3339580253965829) q[3];
U1q(1.4290871319977352, 1.6631465927580007) q[2];
Rz(0.28255045220541325) q[2];
RZZ(1.1370519044424658) q[2], q[3];
U1q(1.4310758013167295, -0.05870573718126487) q[2];
Rz(-0.13294532414044183) q[3];
Rz(-0.07535171856420175) q[2];
RZZ(1.5684865236573335) q[1], q[2];
Rz(-0.1511905917952387) q[1];
U1q(1.4304679870112307, 0.8509980185141269) q[2];
Rz(0.6189273513735188) q[2];
RZZ(1.1372381174921393) q[2], q[3];
U1q(1.4284837092111855, -0.5342943023551773) q[2];
Rz(-0.8305295866908198) q[3];
Rz(-0.42932991480443716) q[2];
if (c[2] == 1) U1q(-1.5707963, 1.5707963) q[3];
if (c[2] == 1) RZZ(1.5707963) q[1], q[3];
if (c[2] == 1) Rz(-1.5707963) q[1];
if (c[2] == 1) U1q(1.5707963, 3.14159265) q[3];
if (c[2] == 1) Rz(-1.5707963) q[3];
if (c[1] == 1) U1q(-1.5707963, 1.5707963) q[2];
if (c[1] == 1) RZZ(1.5707963) q[1], q[2];
if (c[1] == 1) Rz(-1.5707963) q[1];
if (c[1] == 1) U1q(1.5707963, 3.14159265) q[2];
if (c[1] == 1) Rz(-1.5707963) q[2];
Rz(2.6760448241884536) q[1];
U1q(1.5707963, -1.5707963) q[1];
Rz(3.14159265) q[1];
