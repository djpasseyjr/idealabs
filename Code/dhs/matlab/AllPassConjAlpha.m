function [alpha] = AllPassConjAlpha(theta,phi)
%Function for calculating alpha parameter for a discrete all pass filter.

alpha = exp(1i*theta) - exp(.5i*(theta-phi));
end