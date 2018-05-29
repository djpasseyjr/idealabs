%testDiscreteDelta

%{
Transfer Functions:
Basic transfer functions adapted from Vulnerable Links and Secure 
Architectures in the Stabilization of Networks of Controlled Dynamical Systems 

Qbasic Pbasic :
2x2 stable transfer functions from example 1 in the above paper

Qvul Pvul :
3x3 stable transfer functions from fig 5 in the above paper

%}

sampleTime = -1;

Qbasic = tf( {0,1 ; 1,0}, {1,[1,-.2] ; [1,-.4],1},sampleTime);
Pbasic = tf( {1,0 ; 0,1}, {[1,.2],1 ; 1, [1, .2]},sampleTime);

Qvul = tf( {0,1,0 ; 0,0,1 ; 1,0,0},{1,[1,.9],1 ; 1,1,[1,.8] ; [1,.7],1,1},sampleTime);
Pvul = tf( {1,0,0 ; 0,1,0 ; 0,0,1},{[1,.5],1,1 ; 1,[1,.4],1 ; 1,1,[1,.6]},sampleTime);

%Analysis of our dynamic structure functions
Hbasic = inv(eye(size(Qbasic))-Qbasic);
Gbasic = Hbasic*Pbasic;

G = tf({1,1;1,1},{[1,.1+.3i],[1,-.5i];[1,-.2i],[1,-.2i]},1);

phase = [];
theta0 = pi/4;
a = [0+1i,3+2i,2+3i,1+0i,2-1i,1-2i,0-1i,-1-3i,-3 -1i];

for i = 1:10
    z = evalAP(theta0,a(i));
    [p,~] = cart2pol(real(z),imag(z));
    phase(i) = p;
end

phase1 = [];
theta1 = pi/3;


for i = 1:10
     z = evalAP(theta1,a(i));
    [p,~] = cart2pol(real(z),imag(z));
    phase1(i) = p;
end


phase2 = [];
theta2 = pi/2;


for i = 1:10
     z = evalAP(theta2,a(i));
    [p,~] = cart2pol(real(z),imag(z));
    phase2(i) = p;
end


%plot(1:10,phase)

%plot(1:10,phase1)
%plot(1:10,phase2)