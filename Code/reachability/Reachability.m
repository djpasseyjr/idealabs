A = [0,1 ; 0 , 0];
B = [0 ; 1];
C = eye(2,2);
D = [0 ; 0];
SYS = ss(A,B,C,D);


eAT = [1 1; 0 1];

x1 = [1 0]';
x2 = [0 -1]';
x3 = [-1 0]';
x4 = [0 1]';

Wc = [1/3 0.5; 0.5 1];


N = 100; %How many evenly spaced points on the circle to trace with the system

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
    alpha = Wc\(x1 - eAT*x0);
    u1 = zeros(1,res);
    for j = 1:res
        u1(j) = [1 - t(j), 1 ]*alpha;
    end
    U = [ U u1];
    x0 = x1;
end


x1 = Xn(1,:)';
     alpha = Wc\(x1 - eAT*x0);
    u1 = zeros(1,res);
    for j = 1:res
        u1(j) = [1 - t(j), 1 ]*alpha;
    end
    U = [ U u1];
    

t = linspace(0,N,N*res);
y = lsim(SYS,U,t,Xn(1,:)');
clf
plot(y(:,1),y(:,2))
axis equal
grid
