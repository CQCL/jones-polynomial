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
Rz(-0.29317434704860723) q[0];
Rz(1.0408902031258724) q[1];
Rz(0.17157611957759256) q[2];
U1q(1.3802252868099574, 0.7402134432740469) q[1];
Rz(1.3244768447093413) q[1];
RZZ(1.321543080516712) q[1], q[2];
U1q(1.3600932570659898, 0.25548223570118145) q[1];
Rz(0.3442215064678624) q[2];
Rz(1.3069556928138246) q[1];
RZZ(0.5416958444783846) q[0], q[1];
Rz(0.2025587100908538) q[0];
U1q(1.3511371587303405, 0.6830067238346841) q[1];
Rz(0.8846574356582799) q[1];
RZZ(1.3304407443683415) q[1], q[2];
U1q(1.3712746122722297, -0.2337417622383924) q[1];
Rz(0.2811171127441824) q[2];
Rz(0.573514136568065) q[1];
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
Rz(4.100651358044414) q[1];
U1q(1.5707963, -1.5707963) q[1];
Rz(3.14159265) q[1];
