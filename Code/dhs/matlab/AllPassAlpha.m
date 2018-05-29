function [alpha] = AllPassAlpha(theta,phi)
%Second function for computing alpha parameter
alpha = exp(-1i*theta) - exp(-1i*(phi+theta)/2);
end