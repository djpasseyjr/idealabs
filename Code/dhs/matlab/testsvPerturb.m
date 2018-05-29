
%Test for destabilization effectiveness
%{
All = tf();
Faulty = tf();
Exceptional = tf();
Prob = [];
N = 300;
for count=1:N

Results = zeros(4,4);
k = 1;

n = 1;
   G = tf(ones(n,n));
   for i=1:n
       for j=1:n
           a = .7*rand() + .7i*rand();
           b = .7*rand() + .7i*rand();
           G(i,j) = tf([1,-b],[1 -a],1);
           
       end
   end
   
   All = [All,G];
   [Gp,D] = svPerturb(G);
   %Results(1,k) = max(pole(D).*conj(pole(D)))< 1;
   %Results(2,k) = norm(G,inf) > 1;
   Results(3,k) = norm(inv(tf(eye(n,n)) - Gp),inf) - norm(G,inf);
   %Results(4,k) = norm(Gp,inf);
   k = k+1;
   
   %If the destabilization works exceptionally well, store G
   if Results(3,1) >  1000
       Exceptional = [Exceptional,G];
   end
   
   %If the destabilization works poorly, store G
   if Results(3,1) < 10
       Faulty = [Faulty, G];
   end
   
   %Store how effective each destabilization is
   Prob = [Prob , Results(3,1)];

end

failN = [];
excN = [];
allN = [];
[~,ecol] = size(Exceptional);
[~,fcol] = size(Faulty);
for i = 1:ecol
excN(i) = pole(Exceptional(i));
end
for i = 1:fcol
failN(i) = pole(Faulty(i));
end
for i = 1:N
allN(i) = pole(All(i));
end
%}

%Testing higher power proper transfer functions
%{
N = 300;
n = 2;
pertPoles = [];
pertInfNorm = [];
failed = tf();

for i = 1:N
    G = tf(1,[1,-(.7*rand()+.7i*rand())],1);
    for j = 1:n
        F = tf([1,-(.7*rand()+.7i*rand())],[1,-(.7*rand()+.7i*rand())],1);
        G = G*F;
    end
    Gp = svPerturb(G);
    pertPoles(i) = max(pole(inv(tf(1)-Gp)));
    pertInfNorm(i) = norm(inv(tf(1)-Gp),inf) - norm(G,inf);
    if pertInfNorm < norm(G,inf)
        failed = [failed, G];
    end
end
%}

%Test for alpha solver
%{
N= 100;
x = linspace(0,2*pi,N);
[X,Y] = meshgrid(x,x);
Z = zeros(N,N);


for i=1:N
    for j=1:N
        [a,sign,con] = minAlpha(X(i,j),Y(i,j));
        Z(i,j) = norm(a);
    end
end

surf(X,Y,Z)
%}


%Plotting Pole Sizes

N= 100;
x = linspace(-1,1,N);
[X,Y] = meshgrid(x,x);
Z = zeros(N,N);


for i=1:N
    for j=1:N
        %[a,sign,con] = minAlpha(X(i,j),Y(i,j));
        if norm((X(i,j)+1i*Y(i,j))) <= 1
            G = tf(1,[1,(X(i,j)+1i*Y(i,j))],1);
            Gp = G*G'/norm(G,inf)^2;
            P = pole(inv(tf(eye(1))-Gp));
            x = max(sqrt(P.*conj(P)));
            if x >= 10
                Z(i,j) = -1;
            else  
                Z(i,j) = x;
            end
        else
            X(i,j) = NaN;
            Y(i,j) = NaN;
            Z(i,j) = NaN;
        end
        
    end
end

surf(X,Y,Z)
%}
