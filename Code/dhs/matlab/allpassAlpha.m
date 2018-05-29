function alpha = allpassAlpha(theta_0,a,b)


%The above equation is then equivalent to the following system of equations

A = [ a - cos(theta_0) , -b - sin(theta_0) ;
      b - sin(theta_0)  , a + cos(theta_0)  ];
c = [a*cos(theta_0) - b*sin(theta_0)-1, a*sin(theta_0) + b*cos(theta_0)];

x = c/A;

%Solving the above system gives the real and imaginary parts of alpha
alpha = x(1) + 1i*x(2);
end