function theta = frqGtr1(G)
N = 100; %Increasing N does not seem to improve the perturbation;
Theta = linspace(0,2*pi,N);
Mag = 0*(1:N);
for i = 1:N
    Mag(i) = norm(evalfr(G,exp(1i*Theta(i))));
end
[~,i] = max(Mag);
theta = Theta(i);
end