%% Testing how the reconstructions of PnP change with initialisations

% Using the Shepp-Logan phantom
N = 64;
theta = linspace(0,180,257);
theta = theta(1:end-1);
p = N;
d = p-1;
[A,b,x] = paralleltomo(N,theta,p,d);

% Construct noisy data
%seed = 49;
seed = 12;
rng(seed)
noise_level = 0.03;
sigma_train = 10;
e = randn(size(b));
e = noise_level * norm(b) * e / norm(e);
btilde = b + e;

% Regularization parameters
lambda_BM3D = logspace(-4,0,100);
eta = logspace(-6,2,100);

% Use specific lambda and mu from PnPiters.m
indices = 21;
lambda_chosen = lambda_BM3D(indices);
eta_chosen = eta(indices);

% Create number of random initializations of z and u
n_inits = 3;
z_init = zeros(N*N,n_inits);
u_init = zeros(N*N,n_inits);

XBM3D = zeros(N*N,n_inits);
XDnCNN = XBM3D;

opt.display = 1;
opt.tol = 0;
opt.maxiter = 600;

% Stores solutions for each iteration at each noise level
BM3D_x = zeros(N*N,opt.maxiter,n_inits);
DnCNN_x = BM3D_x;

RMSE = zeros(2,opt.maxiter,n_inits);
PSNR = RMSE;

%scale = [1, 1, 1]; % change the scale of the intensities in z and u
scale = [1, 2, 4];

%% Reconstructions

for i = 1:n_inits
    fprintf('Initialization %d out of %d\n',i,n_inits)
    % Generate z and u from a uniform distribution (0,1)
    z_init(:,i) = rand(N*N,1) * scale(i);
    u_init(:,i) = rand(N*N,1) * scale(i);
    opt.z0 = z_init(:,i);
    opt.u0 = u_init(:,i);
    
    [XBM3D(:,i),~,~,infoBM3D] = PnPBM3D(A,btilde,lambda_chosen,1,opt);
    [XDnCNN(:,i),~,~,infoDnCNN] = PnPDnCNN(A,btilde,eta_chosen,sigma_train,opt);
    
    BM3D_x(:,:,i) = infoBM3D.x(:,:);
    DnCNN_x(:,:,i) = infoDnCNN.x(:,:);
    
    for j = 1:opt.maxiter        
        RMSE(1,j,i) = getRMSE(x,BM3D_x(:,j,i));
        RMSE(2,j,i) = getRMSE(x,DnCNN_x(:,j,i));
        PSNR(1,j,i) = getPSNR(x,BM3D_x(:,j,i),1);
        PSNR(2,j,i) = getPSNR(x,DnCNN_x(:,j,i),1);
    end
end

save('PnPInitEst.mat')

%% Load the data

addpath 'Experiments';
load('PnPInitEst.mat')

%% RMSE plots over the iterations

n_iters = 600;

subplot(121)
singlePlot(1:n_iters,RMSE(2,1:n_iters,1),'k',1.2,16,'Iterations','RMSE',RMSE(2,1:n_iters,2),'b',RMSE(2,1:n_iters,3),'r')
title('PnP-DnCNN','fontsize',16)

ax=axes;
set(ax,'units','normalized','position',[0.25,0.7,0.2,0.2])
box(ax,'on')
singlePlot(1:n_iters,RMSE(2,1:n_iters,1),'k',1.2,16,'','',RMSE(2,1:n_iters,2),'b',RMSE(2,1:n_iters,3),'r')
set(ax,'xlim',[0,80],'ylim',[0.0256,0.02565])

subplot(122)
singlePlot(1:n_iters,RMSE(1,1:n_iters,1),'k',1.2,16,'Iterations','RMSE',RMSE(1,1:n_iters,2),'b',RMSE(1,1:n_iters,3),'r')
title('PnP-BM3D','fontsize',16)

%%

load('PnPInitEst3.mat')

%% RMSE plots over the iterations

n_iters = opt.maxiter;

figure (1)
subplot(121)
singlePlot(1:n_iters,RMSE(2,1:n_iters,1),'k',1.2,16,'Iterations','RMSE',RMSE(2,1:n_iters,2),'b',RMSE(2,1:n_iters,3),'r')
title('PnP-DnCNN','fontsize',16)

ax=axes;
set(ax,'units','normalized','position',[0.25,0.7,0.2,0.2])
box(ax,'on')
singlePlot(1:n_iters,RMSE(2,1:n_iters,1),'k',1.2,16,'','',RMSE(2,1:n_iters,2),'b',RMSE(2,1:n_iters,3),'r')
set(ax,'xlim',[0,80],'ylim',[0.0258,0.0265]) % [0.02,0.05]

subplot(122)
singlePlot(1:n_iters,RMSE(1,1:n_iters,1),'k',1.2,16,'Iterations','RMSE',RMSE(1,1:n_iters,2),'b',RMSE(1,1:n_iters,3),'r')
title('PnP-BM3D','fontsize',16)

