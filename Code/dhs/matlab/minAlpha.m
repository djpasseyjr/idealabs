function [alpha,sign,con] = minAlpha(theta,phi)
Phi = [0,2*pi,pi,-pi] + phi;
Alpha = [];

for p=Phi
   Alpha = [Alpha, AllPassAlpha(theta,p), AllPassConjAlpha(theta,p)]; 
end
[~,i] = min((Alpha.*conj(Alpha)).^.5);
alpha = Alpha(i);

sign = 1;
if i > 4
    sign = -1;
end

con = false;
if mod(i,2) == 0
    con = true;
end

end

function [alpha] = AllPassAlpha(theta,phi)
%Second function for computing alpha parameter
alpha = exp(-1i*theta) - exp(-1i*(phi+theta)/2);
end

function [alpha] = AllPassConjAlpha(theta,phi)
%Function for calculating alpha parameter for a discrete all pass filter.
alpha = exp(1i*theta) - exp(1i*(theta-phi)/2);
end
