L = 1;
x = linspace(0,L,241);
t = linspace(0,1,241);
global n p
rng(1)
n_val = 0.5+ rand(1000,1);
p_val = 0.5+ rand(1000,1);
sol = zeros(1000,241,241);
m=0;
for i=1:1000
    n = n_val(i,1);
    p = p_val(i,1);
    sol1 = pdepe(m,@burgerpde,@burgeric,@burgerbc,x,t);
    sol(i,:,:) = sol1;
end

mat_ics =zeros(1000,241,241);
global n p

for i=1:1000
    n = n_val(i,1);
    p = p_val(i,1);
    mat_ics1 = repmat(sin(n*pi*x)+ cos(p*pi*x),241,1);
    mat_ics(i,:,:) = mat_ics1;
end
    
    
save u_sol2_burger.mat t x mat_ics sol