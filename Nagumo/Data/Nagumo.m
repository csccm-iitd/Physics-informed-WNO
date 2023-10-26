
clear
close all
clc

T=1; N=128; epsilon= 0.08;
a=1; J=128; h=a/J; x=(0:h:a)';
alpha = -0.5;

sample = 1000;
mat_ics = zeros(sample, numel(x), N+1);
sol = zeros(sample, numel(x), N+1);
for i =1:sample
    i
    u0= grf(x);    
    [t,ut]=pde_fd(u0,T,a,N,J,epsilon,@(u) u.*(1-u).*(u-alpha),'N');

   mat_ics(i,:,:) = repmat(u0,1,N+1)';
   sol(i,:,:) = ut';
end

% surf(t,x,ut)

save('Nagumo_129_129_1000.mat', 't', 'x', 'sol', 'mat_ics')

function [t,ut]=pde_fd(u0,T,a,N,J,epsilon,fhandle,bctype)
    Dt=T/N; t=[0:Dt:T]'; h=a/J;
    % set matrix A according to boundary conditions
    e=ones(J+1,1); A=spdiags([-e 2*e -e], -1:1, J+1, J+1);
    switch lower(bctype)
        case {'dirichlet','d'}
            ind=2:J; A=A(ind,ind);
        case {'periodic','p'}
            ind=1:J; A=A(ind,ind); A(1,end)=-1; A(end,1)=-1;
        case {'neumann','n'}
            ind=1:J+1; A(1,2)=-2; A(end,end-1)=-2;
    end
    EE=speye(length(ind))+Dt*epsilon*A/h^2;
    ut=zeros(J+1,length(t)); % initialize vectors
    ut(:,1)=u0; u_n=u0(ind); % set initial condition
    for k=1:N % time loop
        fu=fhandle(u_n); % evaluate f(u_n)
        u_new=EE\(u_n+Dt*fu); % linear solve for (1+epsilon A)
        ut(ind,k+1)=u_new; u_n=u_new;
    end
    if lower(bctype)=='p' | lower(bctype)=='periodic'
        ut(end,:)=ut(1,:); % correct for periodic case
    end
end

function [xval] = grf(x) 
    kxx = zeros(numel(x), numel(x));
    sigma = 0.1; l = 0.1;
    for i =1:numel(x)
        for j = 1:numel(x)
            kxx(i,j) = sigma^2*exp(-(x(i)-x(j))^2/(2*l^2));
        end
    end
    xval = mvnrnd(zeros(1,numel(x)), kxx);
    xval = xval';
end
