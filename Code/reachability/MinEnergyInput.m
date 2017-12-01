As = [-1,1 ; 0 , -1];
B = [0 ; 1];
C = eye(2,2);
D = [0 ; 0];
stSYS = ss(As,B,C,D);

Au = [ 0 1 ; 0 0];
usSYS = ss(Au,B,C,D); %Works

%Number of equally spaced points to interpolate on the unit circle
N = 10; 
%Points per period
res = 100;


T = 2*pi/N;

Nvec = 0:N-1;
%Generate the N roots of unity to interpolate
Xn = exp(-1i*Nvec*T);
Xn = [real(Xn)' imag(Xn)' ];

%Xn = fliplr(Xn); %(Optional reverse the order of interpolation)

res = 100;
x0 = Xn(1,:)';
unU = [];
stU = [];
t = linspace(0,T,res);

%Gramian and Exponentiated Matrixes
unstWc = [1/3*(T^3) .5*(T^2); .5*(T^2) T]; %Works
stabWc = [.25*(1-(2*T^2 + 2*T + 1)*exp(-2*T)) .25*(1-(2*T+1)*exp(-2*T)) ; 
    .25*(1-(2*T+1)*exp(-2*T)) .5*(1-exp(-2*T))];

eAT_un = [ 1 T ; 0 1];
eAT_st = exp(-T)*eAT_un;

%UNSTABLE
X0 = [];
Yun = [];
%Generate the inputs for the Unstable System
for i = 2:N
    x1 = Xn(i,:)';
    unAlpha = unstWc\(x1 - eAT_un*x0);
    u1 = zeros(1,res);
    for j = 1:res
        u1(j) = [T - t(j), 1 ]*unAlpha;
    end
    states = lsim(usSYS,u1,t,x0);
    Yun = [Yun ; states];
    x0 = x1;
    
end

x1 = Xn(1,:)';
unAlpha = unstWc\(x1 - eAT_un*x0);
u1 = zeros(1,res);
    for j = 1:res
        u1(j) = [T - t(j), 1 ]*unAlpha;
    end
states = lsim(usSYS,u1,t,x0);
Yun = [Yun ; states];

%STABLE

%Generate inputs for the Stable System
Yst = [];
x0 = Xn(1,:)';

for i = 2:N
    x1 = Xn(i,:)';
    stAlpha = stabWc\(x1 - eAT_st*x0);
    u1 = zeros(1,res);
    for j = 1:res
        u1(j) = exp(t(j) - T)*[T - t(j), 1 ]*stAlpha;
    end
    states = lsim(stSYS,u1,t,x0);
    Yst = [Yst ; states];
    x0= x1;
end


x1 = Xn(1,:)';
    stAlpha = stabWc\(x1 - eAT_st*x0);
    u1 = zeros(1,res);
    for j = 1:res
        u1(j) = exp(t(j) - T)*[T - t(j), 1 ]*stAlpha;
    end
    states = lsim(stSYS,u1,t,x0);
    Yst = [Yst ; states];

    
    
%Plot states

t = linspace(0,2*pi,N*res);

%Plot input with -cos(t) as u(t)
Ucos = -cos(t);
ycos = lsim(usSYS,Ucos,t,Xn(1,:)');

clf
%plot(Yst(:,1),Yst(:,2))
hold on
plot(Yun(:,1),Yun(:,2))
plot(ycos(:,1),ycos(:,2))
axis equal
grid
