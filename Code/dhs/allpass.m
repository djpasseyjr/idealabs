function z = allpass(theta,a)

z = (1-exp(1i*theta)*conj(a))/(exp(1i*theta) - a);