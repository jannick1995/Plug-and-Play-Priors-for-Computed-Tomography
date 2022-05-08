%% Testing how the reconstructions of PnP change over the iterations

% Using the Shepp-Logan phantom
N = 64;
theta = linspace(0,180,257);
theta = theta(1:end-1);
p = N;
d = p - 1;
[A,b,x] = paralleltomo(N,theta,p,d);

% Construct noisy data
seed = 49;
rng(seed)
noise_level = 0.03;
sigma_train = 10;
e = randn(size(b));
e = noise_level * norm(b) * e / norm(e);
btilde = b + e;

% Regularization parameters
lambda_BM3D = logspace(-4,0,100);
eta = logspace(-6,2,100);

% Draw 3 regularization parameters from lambda_BM3D and mu
indices = randi(length(lambda_BM3D),1,3);
lambda_chosen = lambda_BM3D(indices);
eta_chosen = eta(indices);

XBM3D = zeros(N*N,length(lambda_chosen));
XDnCNN = XBM3D;

opt.display = 1;
opt.tol = 0;
opt.maxiter = 1000;

% Stores solutions for each iteration at each noise level
BM3D_x = zeros(N*N,opt.maxiter,length(lambda_chosen));
DnCNN_x = BM3D_x;

RMSE = zeros(2,opt.maxiter,length(lambda_chosen));
PSNR = RMSE;

RMSE_BM3D = zeros(opt.maxiter,length(lambda_chosen));
RMSE_DnCNN = RMSE_BM3D;
PSNR_BM3D = RMSE_BM3D;
PSNR_DnCNN = RMSE_BM3D;

%% Reconstructions

for i = 1:length(lambda_chosen)
    fprintf('Regularization parameter %d out of %d\n',i,length(lambda_chosen))
    [XBM3D(:,i),~,~,infoBM3D] = PnPBM3D(A,btilde,lambda_chosen(i),1,opt);
    [XDnCNN(:,i),~,~,infoDnCNN] = PnPDnCNN(A,btilde,eta_chosen(i),sigma_train,opt);
    
    BM3D_x(:,:,i) = infoBM3D.x(:,:);
    DnCNN_x(:,:,i) = infoDnCNN.x(:,:);
    
    for j = 1:opt.maxiter
        RMSE_BM3D(j,i) = getRMSE(x,BM3D_x(:,j,i));
        RMSE_DnCNN(j,i) = getRMSE(x,DnCNN_x(:,j,i));
        PSNR_BM3D(j,i) = getPSNR(x,BM3D_x(:,j,i),1);
        PSNR_DnCNN(j,i) = getPSNR(x,DnCNN_x(:,j,i),1);
        
        RMSE(1,j,i) = getRMSE(x,BM3D_x(:,j,i));
        RMSE(2,j,i) = getRMSE(x,DnCNN_x(:,j,i));
        PSNR(1,j,i) = getPSNR(x,BM3D_x(:,j,i),1);
        PSNR(2,j,i) = getPSNR(x,DnCNN_x(:,j,i),1);
    end
end

save('PnPiters.mat')

%% Loadthe data

load('PnPiters.mat')

%% RMSE plots over the iterations

n_iters = 1000;

subplot(121)
singlePlot(1:n_iters,RMSE(2,1:n_iters,1),'k',1.2,16,'Iterations','RMSE',RMSE(2,1:n_iters,2),'r',RMSE(2,1:n_iters,3),'b')
legend({join(['$\eta=$ ',num2str(eta_chosen(1),3)]),join(['$\eta=$ ',num2str(eta_chosen(2),3)]),join(['$\eta=$ ',num2str(eta_chosen(3),3)])},'fontsize',14)
title('PnP-DnCNN','fontsize',16)

% ax=axes;
% set(ax,'units','normalized','position',[0.25,0.7,0.2,0.2])
% box(ax,'on')
% singlePlot(1:n_iters,RMSE(2,1:n_iters,1),'k',1.2,16,'','',RMSE(2,1:n_iters,2),'b',RMSE(2,1:n_iters,3),'r')
% set(ax,'xlim',[0,40])


subplot(122)
singlePlot(1:n_iters,RMSE(1,1:n_iters,1),'k',1.2,16,'Iterations','RMSE',RMSE(1,1:n_iters,2),'r',RMSE(1,1:n_iters,3),'b')
ylim([0,0.2])
legend({join(['$\lambda=$ ',num2str(lambda_chosen(1),3)]),join(['$\lambda=$ ',num2str(lambda_chosen(2),3)]),join(['$\lambda=$ ',num2str(lambda_chosen(3),3)])},'location','northwest','fontsize',14)
title('PnP-BM3D','fontsize',16)
% ax=axes;
% set(ax,'units','normalized','position',[0.7,0.7,0.2,0.2])
% box(ax,'on')
% singlePlot(1:n_iters,RMSE(1,1:n_iters,1),'k',1.2,16,'','',RMSE(1,1:n_iters,2),'b',RMSE(1,1:n_iters,3),'r')
% set(ax,'xlim',[0,40])

%% Reconstructions over different iterations

idx = [10, 100, 500, 1000];
%idx = [1, 2, 4, 8];

% Fontsize is huge because the figure needs to be in full size!

figure (1)
[ha, pos] = tight_subplot(2,4,0.01,0.1,0.03);
for i = 1:length(idx)
    axes(ha(i))
    imageplot(reshape(BM3D_x(:,idx(i),3),N,N),[-0.1,1.1])
    title(sprintf('%d iterations',idx(i)),'fontsize',30)
    if i == 4
        colorbar
        set(ha(i),'position',pos{i})
    end
    
    axes(ha(i+4))
    imageplot(reshape(DnCNN_x(:,idx(i),3),N,N),[-0.1,1.1])
    if i+4 == 8
        colorbar
        set(ha(i+4),'position',pos{i+4})
    end
end

