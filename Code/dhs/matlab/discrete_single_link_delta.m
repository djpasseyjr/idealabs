function [delta, G2] = discrete_single_link_delta(P, Q, Hji, j, i)
%Inputs and Input parameters: P,Q are mimo transfer function objects, Hji
%is a siso transfer function object, P and Q are used to calculate the
%final G2. j and i are integers reffering to the col and row respectivly of
%Q that delta should be added to as a perturbation.
%Outputs: delta: a siso transfer function representing the pertubation to
%add to Qij.
%Summary: This function calculates a perturbation with a norm slightly
%larger than 1/||Hji||inf that when added to Qij will destabalize the whole
%system G = (I-Q)*P
[a, ~] = size(Q.num);
theta_0 = sqrt(2)/2 + sqrt(2)/2*i;
[U,S,V] = svd(evalfr(Hji, theta_0)); % Could also be done 
% as H = evalfr(Hji, 0.1i), S = (real(H)^2+imag(H)^2)^.5, U = H/S, V = 1
sig_inv = 1/S;
H = evalfr(Hji, theta_0);
% We rewrite the u and v in terms of polar coordinates
[vphi, vroe] = cart2pol(real(V'), imag(V')); % in python:  z2polar = lambda z: ( np.abs(z), np.angle(z) ), rS, thetaS = z2polar( z ), from: "http://stackoverflow.com/questions/20924085/python-conversion-between-coordinates"
[uphi, uroe] = cart2pol(real(U'), imag(U'));
% Here we make sure that theta is between 0 and pi
%if vtheta < 0
%    vroe = -1*vroe;
%    vtheta = pi + vtheta;
%end
%if utheta < 0
%    uroe = -1*uroe;
%    utheta = pi + utheta;
%end

% We calculate a function whose phase equals the given values
% Put in the solution for alpha... don't use the lines below up to 41...
% Put in the discrete time all pass filter as well.

%TRANSFER FUNCTION FOR V:
alpha = 2 + 2*i; 
%This is a arbitrary fixing. Alpha is a dummy parameter that we use 
%because of discontinuities in the all pass filter. 
                  
quotient = exp(vphi*i)*(exp(theta_0*i) - alpha) / (1. - exp(theta_0*i)*conj(alpha));
Rquot = real(quotient);
Imquot = imag(quotient);

%We need to pick a convention for our fourier inputs. I'm using
%e^i*theta

A = [Rquot - cos(theta_0), -1*sin(theta_0) - Imquot;
     Imquot - sin(theta_0), cos(theta_0) + Rquot  ];

b = [Rquot*cos(theta_0) - Imquot*sin(theta_0)-1, Rquot*sin(theta_0) + Imquot*cos(theta_0)];
x = A/b;

beta_v = x(1) + x(2)*i;

v = tf(vroe*[1 ,-1*conj(alpha+beta_v), conj(alpha*beta_v)],[alpha*beta_v,-1*(alpha+beta_v),1]);
%make sure that these are in the correct order. I checked but we should
%double check.

%TRANSFER FUNCTION FOR U
quotient = exp(uphi*i)*(exp(theta_0*i) - alpha) / (1. - exp(theta_0*i)*conj(alpha));
Rquot = real(quotient);
Imquot = imag(quotient);

%We need to pick a convention for our fourier inputs. I'm using
%e^i*theta because it was easier to do algebra with.

A = [Rquot - cos(theta_0), -1*sin(theta_0) - Imquot;
     Imquot - sin(theta_0), cos(theta_0) + Rquot  ];

b = [Rquot*cos(theta_0) - Imquot*sin(theta_0)-1, Rquot*sin(theta_0) + Imquot*cos(theta_0)];
x = A/b;

beta_u = x(1) + x(2)*i;

u = tf(uroe*[1 ,-1*conj(alpha+beta_u), conj(alpha*beta_u)],[alpha*beta_u,-1*(alpha+beta_u),1]);
%make sure that these are in the correct order.


%CALCULATE DELTA
delta = sig_inv * vroe * uroe * v * u;

%alpha = (1/norm(Hji, inf)) / norm(delta, inf); %To lower delta

Q(i, j) = Q(i, j) + delta %* alpha;
G2 = inv(eye(a) - Q)*P;
delta = delta %* alpha;
%{
alpha = max(abs(pole(G2))); % Make the pertubation smaller
Q(i, j) = Q(i, j) - delta;
delta = delta / alpha;
Q(i, j) = Q(i, j) + delta;
G2 = inv(eye(a) - Q)*P;

%}

