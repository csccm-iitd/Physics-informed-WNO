function [c,f,s] = burgerpde(x,t,u,dudx)
c = 1;
f = 0.1*dudx;
s = -u*dudx;
end