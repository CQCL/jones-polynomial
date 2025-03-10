OPENQASM 2.0;

opaque U1q(p0, p1) q0;
opaque Rz(p0) q0;
opaque RZZ(p0) q0, q1;

qreg q[5];
creg c[4];

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
if (c[3] == 1) U1q(-1.5707963, 1.5707963) q[4];
if (c[3] == 1) RZZ(1.5707963) q[1], q[4];
if (c[3] == 1) Rz(-1.5707963) q[1];
if (c[3] == 1) U1q(1.5707963, 3.14159265) q[4];
if (c[3] == 1) Rz(-1.5707963) q[4];
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
Rz(1.528566355970471) q[2];
Rz(1.6577977198621565) q[3];
Rz(0.7834061641827169) q[4];
U1q(1.3341181095212369, -0.5511855944838482) q[3];
Rz(1.680503296696676) q[3];
RZZ(0.9428378148577639) q[3], q[4];
U1q(-1.3130139897132735, 2.074731243938526) q[3];
Rz(1.1198268286046515) q[4];
Rz(2.9117118468553875) q[3];
RZZ(1.2501420384439996) q[2], q[3];
Rz(2.2365498605554595) q[2];
U1q(-1.3133786816759572, -0.708750504335997) q[3];
Rz(0.9371838785103748) q[3];
RZZ(0.94267131508963) q[3], q[4];
U1q(1.3345161387790823, 1.1738038880731305) q[3];
Rz(1.2620801327935334) q[4];
Rz(1.0346010554622622) q[3];
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
Rz(1.6788048119099352) q[2];
Rz(1.389184869822185) q[3];
Rz(0.9388801590056095) q[4];
U1q(1.4929690501868802, -0.02146858359059794) q[3];
Rz(1.3879946366092946) q[3];
RZZ(1.1208015329361796) q[3], q[4];
U1q(1.493927274077808, -0.6561788535491784) q[3];
Rz(0.6944379926432168) q[4];
Rz(1.4033426127718334) q[3];
RZZ(1.5670097865191353) q[2], q[3];
Rz(1.7739379706067475) q[2];
U1q(-1.492562923475675, -1.1330286067647313) q[3];
Rz(0.8294342552805627) q[3];
RZZ(1.121033391933071) q[3], q[4];
U1q(1.4917484960326726, 0.8154828926955129) q[3];
Rz(0.5617571120766232) q[4];
Rz(0.5980743505425773) q[3];
if (c[3] == 1) U1q(-1.5707963, 1.5707963) q[4];
if (c[3] == 1) RZZ(1.5707963) q[1], q[4];
if (c[3] == 1) Rz(-1.5707963) q[1];
if (c[3] == 1) U1q(1.5707963, 3.14159265) q[4];
if (c[3] == 1) Rz(-1.5707963) q[4];
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
Rz(2.8780560126064683) q[1];
U1q(1.5707963, -1.5707963) q[1];
Rz(3.14159265) q[1];
