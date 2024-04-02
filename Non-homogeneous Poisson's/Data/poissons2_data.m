rng(1)
n_grid_s = 65;
A_val = -2 + 4*rand(1000,1);
B_val = -2 + 4*rand(1000,1);
sol = zeros(1000,n_grid_s,n_grid_s);
mat_sd = zeros(1000,n_grid_s,n_grid_s);
for k=1:1000
    A = A_val(k,1);
    B = B_val(k,1);
    x1d=linspace(-1,1,n_grid_s);
    y1d=linspace(-1,1,n_grid_s);
    [x,y]= meshgrid(x1d,y1d);
    zd = zeros(n_grid_s,n_grid_s);
    sd = zeros(n_grid_s,n_grid_s);
    for i=1:n_grid_s
        for j=1:n_grid_s
            zd(i,j) = A*sin(pi*x1d(j))*(1+cos(pi*y1d(i)))+ B*sin(2*pi*x1d(j))*(1-cos(2*pi*y1d(i)));
            sd(i,j) = 4*B*pi^2*cos(2*pi*y1d(i))*sin(2*pi*x1d(j)) - A*pi^2*cos(pi*y1d(i))*sin(pi*x1d(j)) - A*pi^2*sin(pi*x1d(j))*(cos(pi*y1d(i)) + 1) + 4*B*pi^2*sin(2*pi*x1d(j))*(cos(2*pi*y1d(i)) - 1);
        end
    end
   sol(k,:,:) = zd;
   mat_sd(k,:,:) = sd;
end
save u_sol_poissons.mat x1d y1d mat_sd sol
