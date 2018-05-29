function [Gp,delta] = svPerturb(G)

%Find peak of G

[~,theta] = norm(G,inf);
%Uncomment for more Arbitrary theta
%theta = frqGtr1(G);


%Evaluate G at it's peak
[U,S,V] = svd(evalfr(G,exp(1i*theta)));

%Set variables for delta construction
sampleT = G.Ts;
sigma = 1/S(1,1);
u = U(:,1);
v = V(:,1);
%}
%{
z = evalfr(G,exp(1i*theta));
[phi,r] = cart2pol(real(z),imag(z));
sigma = 1./r;
u = exp(phi*1i);
v = 1;
%}

%Create unitary transfer function vectors
vtf = allPassTfVec(v,theta,true);
utf = allPassTfVec(u,theta,false);

%Compute delta and perturb G
delta = sigma*transpose(vtf)*transpose(utf');

%Uncomment for Frequency Invariant Perturbation:
%delta = tf(sigma*v*u');
Gp = G*delta;

end

function vtf = allPassTfVec(v,theta,V)
[vlen,~] = size(v);
vtf = tf();
    %The following loop fills the transfer function entry by entry:
    for k = 1:vlen
        z = v(k,1);
        [phi,r] = cart2pol(real(z),imag(z));
        vtf(k) = r*poleMinimizingFilter(theta,phi,V);
    end


end

function filter = poleMinimizingFilter(theta,phi,V)

alpha = AllPassAlpha(theta,phi);
lessth1 = norm(alpha)<1;

if ~V
    lessth1 = ~lessth1;
end

if lessth1
    filter = exp(2i*phi)*tf([-alpha,1],[1,-conj(alpha)],1);
else
    filter = tf([1,-conj(alpha)],[-alpha,1],1);
end

end

function [alpha] = AllPassAlpha(theta,phi)
%Second function for computing alpha parameter
alpha = exp(-1i*theta) - exp(-1i*(phi+theta)/2);
end