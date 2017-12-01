function [G2,delta] = svPerturb(G)

%Find peak of G
[~,theta_0] = norm(G,inf);

%Evaluate G at it's peak
[U,S,V] = svd(evalfr(G,exp(1i*theta_0)));

%Set variables for delta construction
sampleT = G.Ts;
sigma = 1/S(1,1);
u = U(:,1);
v = V(:,1);
[ulen,~] = size(u);
[vlen,~] = size(v);

alpha = 2 + 2*1i; 
%This is a arbitrary fixing. Alpha is a dummy parameter that we use 
%because of discontinuities in the all pass filter. As long alpha is not
%unitary, any alpha will do.

%make empty transfer functions
vtf = tf();
utf = tf();


%UNITARY TRANSFER FUNCTION FOR v VECTOR:


for k = 1:vlen
    %The following loop fills the transfer function entry by entry:

    z = v(k,1);
    [vphi, vroe] = cart2pol(real(z), imag(z));
    
    %Solve for beta parameter

    beta = AllPassBeta(theta_0,vphi,alpha);

    %Create unitary all pass filter with the beta and alpha parameters
    vtf(k) = vroe*tf([-conj(beta),1],[1,-beta])*tf([-conj(alpha),1],[1,-alpha]);
    
end


%UNITARY TRANSFER FUNCTION FOR u VECTOR:
for k = 1:ulen
    z = u(k,1);
    %Transform each entry of u to polar coordinates
    [uphi, uroe] = cart2pol(real(z), imag(z));
    
    %Solve for beta parameter
    beta = AllPassBeta(theta_0,uphi,alpha);

    %create unitary all pass filter with beta and alpha parameters
    utf(k) = uroe*tf([-conj(beta),1],[1,-beta])*tf([-conj(alpha),1],[1,-alpha]);

end

%Compute delta and perturb G
delta = sigma*(vtf'*utf);
delta.Ts = sampleT;

G2 = G*delta;

end


function [beta,quotient] = AllPassBeta(theta_0,phi,alpha)
%{
Function for calculating beta parameter for a discrete all pass filter.

Let AP_beta(z) and AP_alpha(z) be discrete all pass filters. Then given
theta_0,alpha and phi, this fuction solves for beta such that:

e^(i*phi)/AP_alpha(e^(i*theta_0)) = AP_beta(e^(i*theta_0))

%}

%Compute left side of equation and separate real and imaginary parts.
quotient = exp(1i*phi)*(exp(1i*theta_0) - alpha) / (1 - exp(1i*theta_0)*conj(alpha));
a = real(quotient);
b = imag(quotient);
 norm(a+1i*b)
%The above equation is then equivalent to the following system of equations

A = [ a - cos(theta_0) , -b - sin(theta_0) ;
      b - sin(theta_0)  , a + cos(theta_0)  ];
c = [a*cos(theta_0) - b*sin(theta_0)-1, a*sin(theta_0) + b*cos(theta_0)];

x = c/A;

%Solving the above system gives the real and imaginary parts of beta

beta = x(1) + 1i*x(2);


end