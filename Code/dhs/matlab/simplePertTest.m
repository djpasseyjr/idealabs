N = 100;
r = 4;
n = 3;
m = 5;
Work = [];
NotWork = [];
Wpeak =[];
NWpeak = [];
for i = 1:N
    H = tf(poly(-2*rand([1,n])+1),poly(-2*rand([1,m])+1),1);
    [N,peak] = norm(H,inf);
    if abs(real(exp(1i*peak))) ~= 1
        NotWork = [NotWork H];
        NWpeak = [NWpeak, peak];
    else
        Wpeak = [Wpeak,peak];
        Work = [Work,H];
    end
end

%{
%Graph the pole distribution of working tfs and not working
Wp = pole(Work);
NWp = pole(NotWork);
y = [ones([1,length(Wp)]),1.1*ones([1,length(NWp)]),-1,2];
poles = [Wp' NWp' 1 1];
sz = 25;
c = [linspace(1,1,length(Wp)),linspace(15,15,length(NWp)),0,0];
scatter(poles,y,sz,c,'filled')
%}