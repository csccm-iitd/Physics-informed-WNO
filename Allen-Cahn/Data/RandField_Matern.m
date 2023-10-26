function [F]= RandField_Matern(lam_x,lam_y,nu,var,align,k,vis)
%% This routine generate random field in 2D using non-isotropic matern covariance function
% % References

% % Dietrich, C. and Newsam, G., Fast and exact simulation of stationary Gaussian processes
% % through circulant embedding of the covariance matrix, SIAM J. Sci. Comput., 18:1088–1107, 1997.

% % Examples
% [F]= RandField_Matern(0.1,0.1,1,1,0, 7,1); % Isotropic
% [F]= RandField_Matern(2,0.02,0.5,1,0,7,1); % layering along x
% [F]= RandField_Matern(0.01,1,0.5,0.5,0,7,1); % layering along y
% [F]= RandField_Matern(0.01,1,0.5,0.5,pi/4,7,1); % aligned 45 degrees with x-axis

% % Inputs parameters
% % lam_x (positive)=correlation length along x-direction
% % lam_y (positive)=correlation length along y-direction
% % nu (>=0)= smoothness
% % var(>0)= variance of the covariance function
% % align = alignment with horizontal axis, for example align = pi/4
% % 2^k x 2^k (k= integer) = grid size
% % vis (0 or 1) = visualization of the random field

% % output is F=2^k x2^k matrix with the random field

% % IMPORTANT NOTE: The algorithm may take long time to finish for very large correlation lengths. Complexity is typically NlogN, N=2^2k

% rho defines a parametrized Matern function for covariance, can be replaced with other
% covariance functions...

% rho = @(h)var*(2^(1-nu)/gamma(nu))*((2*sqrt(nu)*(sqrt((h(1)/lam_x)^2+(h(2)/lam_y)^2))/(2^(k))).^nu).*...
%     besselk(nu,(2*sqrt(nu)*(sqrt((h(1)/lam_x)^2+(h(2)/lam_y)^2))/(2^(k))));

rho = @(index)var*(2^(1-nu)/gamma(nu))*((2*sqrt(nu)*(sqrt(((index(1)*cos(align)+index(2)*sin(align))/lam_x)^2+((-index(1)*sin(align)+index(2)*cos(align))/lam_y)^2))/(2^(k))).^nu).*...
            besselk(nu,(2*sqrt(nu)*(sqrt(((index(1)*cos(align)+index(2)*sin(align))/lam_x)^2+((-index(1)*sin(align)+index(2)*cos(align))/lam_y)^2))/(2^(k))));

m = 2^k;
n = 2^k;

%% lam needs to be computed only once, so save it if many random fields are required to be generated
lam = stationary_Gaussian_process(m,n,rho,var);

%% Next three steps generate a sample of random field using lam
F = fft2(lam.*complex(randn(size(lam)),randn(size(lam))));
F = F(1:m+1,1:n+1);
F=real(F);

%% Visualization
if(vis==1)
    imagesc(real(F))
     colormap(jet)
     colorbar EastOutside
end
return