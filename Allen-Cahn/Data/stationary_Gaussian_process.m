function [lam]=stationary_Gaussian_process(m,n,rho,var)
%% This perform factorization of the Covariance matrix

%%% Reference:
% Kroese, D. P., & Botev, Z. I. (2015). Spatial Process Simulation.
% In Stochastic Geometry, Spatial Statistics and Random Fields(pp. 369-404)
% Springer International Publishing, DOI: 10.1007/978-3-319-10064-7_12

padd=0;
nnev=0;
while(padd==0)
    padd=1;
    tx=0:n-1;
    ty=0:m-1;
    Rows=zeros(m,n); Cols=Rows;
    for i=1:n % sample covariance function at grid points;
        for j=1:m
            Rows(j,i)=rho([tx(i)-tx(1),ty(j)-ty(1)]); % rows of blocks of cov matrix
            Cols(j,i)=rho([tx(1)-tx(i),ty(j)-ty(1)]); % columns of blocks of cov matrix
            
        end
    end
    Cols(1,1)=var;
    Rows(1,1)=var;
    % create the first row of the block circulant matrix with circular blocks
    % and store it as a matrix suitable for fft2;
    
    BCR=[Rows, Cols(:,end:-1:2);
        Cols(end:-1:2,:), Rows(end:-1:2,end:-1:2)];
    % compute eigenvalues
    
    lam=real(fft2(BCR))/(2*m-1)/(2*n-1);
    if abs(min(lam(lam(:)<0)))>10^-15
        padd=0;
        nnev=ceil(log2(nnz(lam)));
        
    else
        lam(lam(:)<0)=0; lam=sqrt(lam);
    end
    
    m=m+nnev;
    n=n+nnev;
end
return