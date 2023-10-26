clear
close all
clc

T= 5; N=104; a=[3 3]; k= 6; J=[2^k 2^k]; epsilon=1;
x=(0:a(1)/J(1):a(1))'; y=(0:a(2)/J(2):a(2))';
% % [xx yy]=meshgrid(y,x); 
% % u0=sin(xx).*cos(pi*yy/8);

% Navaneeth data
% T= 1; N=50; a=[1 1]; k= 6; J=[2^k 2^k]; epsilon= 0.01;
% x=(0:a(1)/J(1):a(1))'; y=(0:a(2)/J(2):a(2))';

% T= 1; N=100; a=[1 1]; k= 6; J=[2^k 2^k]; epsilon= 0.1;
% x=(0:a(1)/J(1):a(1))'; y=(0:a(2)/J(2):a(2))';

% [xx, yy]=meshgrid(y,x); 
% u00=sin(xx).*cos(4*pi*yy);

sample = 1;
initial = zeros(numel(x), numel(y), sample);
sol = zeros(numel(x), numel(y), N+1, sample);
for i =1:sample
    i
    u0= RandField_Matern(0.1, 0.1, 5, 0.1, 0, k, 1);
    [t,ut]=pde_twod_Gal(u0,T,a,N,J,epsilon,@(u) u-u.^3);

    initial(:,:,i) = u0;
    sol(:,:,:,i) = ut;
end

% save('Allen_Cahn_pde_65_65_800.mat', 't', 'x', 'sol', 'initial')

%
figure(1); imagesc(u0); colormap(jet)

figure(1)
index = 1;
for i = 1:size(t,1)
    if mod(i,2)==0
        subplot(5,5,index); imagesc(sol(:,:,i)); colorbar();
        index = index+1;
    end
end

% figure(2)
% for i = 1: size(t,1)
%     clf
%     imagesc(ut(:,:,i)); colormap(jet); colorbar(); ylim([0,65]);
%     pause(0.1)
% end

function [t,ut]=pde_twod_Gal(u0,T,a,N,J,epsilon,fhandle)
    Dt=T/N; t=[0:Dt:T]'; ut=zeros(J(1)+1,J(2)+1,N);
    % set linear operators
    lambdax=2*pi*[0:J(1)/2 -J(1)/2+1:-1]'/a(1);
    lambday=2*pi*[0:J(2)/2 -J(2)/2+1:-1]'/a(2);
    [lambdaxx, lambdayy]=meshgrid(lambday,lambdax);
    M=epsilon*(lambdaxx.^2+lambdayy.^2); EE=1./(1+Dt*M);
    ut(:,:,1)=u0; u=u0(1:J(1),1:J(2)); uh=fft2(u); % set initial data
    for n=1:N, % time loop
        fhu=fft2(fhandle(u)); % compute fhat
        uh_new=EE.*(uh+Dt*fhu);
        u=real(ifft2(uh_new)); ut(1:J(1), 1:J(2), n+1)=u; uh=uh_new;
    end
ut(J(1)+1,:,:)=ut(1,:,:); ut(:,J(2)+1,:)=ut(:,1,:); % make periodic
end

