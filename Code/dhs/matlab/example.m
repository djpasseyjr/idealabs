A = [-1,1 ; -1 , 0];
B = [0 ; 1];
C = eye(2,2);
D = [0 ; 0];
SYS = ss(A,B,C,D);

%{
eAT = [1 1; 0 1];

x1 = [1 0]';
x2 = [0 -1]';
x3 = [-1 0]';
x4 = [0 1]';

Wc = [1/3 0.5; 0.5 1];
alpha1 = inv(Wc)*(x2 - eAT*x1);
alpha2 = inv(Wc)*(x3 - eAT*x2);
alpha3 = inv(Wc)*(x4 - eAT*x3);
alpha4 = inv(Wc)*(x1 - eAT*x4);

SYS1 = ss(A,B,C,D);
N = 500;
t = linspace(0,1,N)';
for i = 1:N
    u1(i) = [1-t(i) 1]*alpha1;
    u2(i) = [1-t(i) 1]*alpha2;
    u3(i) = [1-t(i) 1]*alpha3;
    u4(i) = [1-t(i) 1]*alpha4;
end
u = [u1'; u2'; u3'; u4']

t = linspace(0,4,4*N);
y = lsim(SYS1,u,t,[0;0]);
clf
plot(y(:,1),y(:,2))
grid
%}



N = 4; %How many evenly spaced points on the circle to trace with the system
Wc = [.25*(1-5*exp(-2)) .25*(1-3*exp(-2)); 0.25*(1-3*exp(-2)) .5*(1-exp(-2))]; %Controlability gramian
eAT = exp(-1)*[1 1; 0 1];   %System Matrix solution for t=1 u=o

Nvec = 0:N-1;
Xn = exp(-1i*Nvec*2*pi/N);
Xn = [real(Xn)' imag(Xn)' ];
%Xn = fliplr(Xn);

res = 1000;
x0 = Xn(1,:)';
U = [];
t = linspace(0,1,res);
for i = 2:N
    x1 = Xn(i,:)';
    alpha = inv(Wc)*(x1 - eAT*x0);
    u1 = zeros(1,res);
    for j = 1:res
        u1(j) = exp(t(i)-1)*[1 - t(j), 1 ]*alpha;
    end
    U = [ U u1];
    x0 = x1;
end


x1 = Xn(1,:)';
    alpha = inv(Wc)*(x1 - eAT*x0);
    u1 = zeros(1,res);
    for j = 1:res
        u1(j) = exp(t(i)-1)*[1 - t(j), 1 ]*alpha;
    end
    U = [ U u1];
    

t = linspace(0,N,N*res);
y = lsim(SYS,U,t,Xn(1,:)');
clf
plot(y(:,1),y(:,2))
axis equal
grid
