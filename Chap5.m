%% Figures for Image Denoising Chapter

% Add path to the needed algorithms
addpath 'TVregADMM';
addpath 'DnCNN/model/specifics/';
addpath 'DnCNN/utilities/'; 
addpath 'BM3D/bm3d/';

% addpath 'TVregADMM';
% addpath 'DnCNN/model/specifics/'
% addpath 'DnCNN/utilities/'; 
% addpath 'BM3D/bm3d/';

% Noisy phantom considered throughout the chapter
N = 64;
X = phantom(N);

seed = 78;
rng(seed)
noise_level = 0.08;
e = randn(size(X(:)));
e = noise_level * norm(X(:)) * e / norm(e);
Xtilde = reshape(X(:)+e,N,N);

%% Median filtering

% Using different sized windows
Xmed = medfilt2(Xtilde);
Xmed2 = medfilt2(Xtilde,[2,2]);

setPlot()
figure (1)
[ha, pos] = tight_subplot(2,2,0.06,0.06,0.1);
axes(ha(1))
imageplot(X,[-0.1,1.1])
colorbar
set(ha(1),'position',pos{1})
title('$||e||_2/||b||_2=0$','fontsize',16)
axes(ha(2))
imageplot(Xtilde,[-0.1,1.1])
colorbar
set(ha(2),'position',pos{2})
title('$||e||_2/||b||_2=0.08$','fontsize',16)
axes(ha(3))
imageplot(Xmed,[-0.1,1.1])
colorbar
set(ha(3),'position',pos{3})
title('$3\times 3$ Median filter','fontsize',16)
axes(ha(4))
imageplot(Xmed2,[-0.1,1.1])
colorbar
set(ha(4),'position',pos{4})
title('$2\times 2$ Median filter','fontsize',16)

%% TV denoising (using TVregADMM)

opt.abstol = 1e-3;
opt.reltol = 1e-3;

XTV1 = TVregADMM(eye(N*N,N*N),Xtilde(:),0.02,1,opt);
XTV2 = TVregADMM(eye(N*N,N*N),Xtilde(:),0.11,1,opt);

figure (2)
[ha, pos] = tight_subplot(2,2,0.06,0.06,0.1);
axes(ha(1))
imageplot(X,[-0.1,1.1])
colorbar
set(ha(1),'position',pos{1})
title('$||e||_2/||b||_2=0$','fontsize',16)
axes(ha(2))
imageplot(Xtilde,[-0.1,1.1])
colorbar
set(ha(2),'position',pos{2})
title('$||e||_2/||b||_2=0.08$','fontsize',16)
axes(ha(3))
imageplot(reshape(XTV1,N,N),[-0.1,1.1])
colorbar
set(ha(3),'position',pos{3})
title('TV denoising $\lambda=0.02$','fontsize',16)
axes(ha(4))
imageplot(reshape(XTV2,N,N),[-0.1,1.1])
colorbar
set(ha(4),'position',pos{4})
title('TV denoising $\lambda=0.11$','fontsize',16)

%% BM3D denoising

figure (3)
[ha, pos] = tight_subplot(2,2,0.06,0.06,0.1);
axes(ha(1))
imageplot(X,[-0.1,1.1])
colorbar
set(ha(1),'position',pos{1})
title('$||e||_2/||b||_2=0$','fontsize',16)
axes(ha(2))
imageplot(Xtilde,[-0.1,1.1])
colorbar
set(ha(2),'position',pos{2})
title('$||e||_2/||b||_2=0.08$','fontsize',16)
axes(ha(3))
imageplot(BM3D(Xtilde,0.024),[-0.1,1.1]) % 0.024
colorbar
set(ha(3),'position',pos{3})
title('BM3D $\sigma_e=0.024$','fontsize',16)
axes(ha(4))
imageplot(BM3D(Xtilde,0.1),[-0.1,1.1])
colorbar
set(ha(4),'position',pos{4})
title('BM3D $\sigma_e=0.1$','fontsize',16)

%% DnCNN denoising

% Choose trained denoising strength
sigma = [10, 15];

% Construct residual mappings 
resMaps = zeros(N,N,length(sigma));
DnCNNdenoised = resMaps;

for i = 1:length(sigma)
    % Load the correct network
    load(fullfile('model','specifics',['sigma=',num2str(sigma(i),'%02d'), ...
    '.mat']),'net');
    residualImage = simplenn_matlab(net,mat2gray(Xtilde));
    resMaps(:,:,i) = residualImage(end).x;
    DnCNNdenoised(:,:,i) = Xtilde - resMaps(:,:,i);
end

figure (4)
[ha, pos] = tight_subplot(2,2,0.06,0.06,0.1);
axes(ha(1))
imageplot(X,[-0.1,1.1])
colorbar
set(ha(1),'position',pos{1})
title('$||e||_2/||b||_2=0$','fontsize',16)
axes(ha(2))
imageplot(Xtilde,[-0.1,1.1])
colorbar
set(ha(2),'position',pos{2})
title('$||e||_2/||b||_2=0.08$','fontsize',16)
axes(ha(3))
imageplot(DnCNNdenoised(:,:,1),[-0.1,1.1])
colorbar
set(ha(3),'position',pos{3})
title('DnCNN $\sigma_e=10/255$','fontsize',16)
axes(ha(4))
imageplot(DnCNNdenoised(:,:,2),[-0.1,1.1])
colorbar
set(ha(4),'position',pos{4})
title('DnCNN $\sigma_e=15/255$','fontsize',16)

%% Considering a noisier phantom for DnCNN

N = 64;
X = phantom(N);

seed = 78;
rng(seed)
noise_level = 0.20; 
e = randn(size(X(:)));
e = noise_level * norm(X(:)) * e / norm(e);
Xtilde = reshape(X(:)+e,N,N);

% Choose trained denoising strength
sigma = [10, 15];

% Construct residual mappings 
resMaps = zeros(N,N,length(sigma));
DnCNNdenoised = resMaps;

for i = 1:length(sigma)
    % Load the correct network
    load(fullfile('model','specifics',['sigma=',num2str(sigma(i),'%02d'), ...
    '.mat']),'net');
    residualImage = simplenn_matlab(net,mat2gray(Xtilde));
    resMaps(:,:,i) = residualImage(end).x;
    DnCNNdenoised(:,:,i) = Xtilde - resMaps(:,:,i);
end

figure (4)
[ha, pos] = tight_subplot(2,2,0.06,0.06,0.1);
axes(ha(1))
imageplot(X,[-0.1,1.1])
colorbar
set(ha(1),'position',pos{1})
title('$||e||_2/||b||_2=0$','fontsize',16)
axes(ha(2))
imageplot(Xtilde,[-0.1,1.1])
colorbar
set(ha(2),'position',pos{2})
title('$||e||_2/||b||_2=0.2$','fontsize',16)
axes(ha(3))
imageplot(DnCNNdenoised(:,:,1),[-0.1,1.1])
colorbar
set(ha(3),'position',pos{3})
title('DnCNN $\sigma_e=10/255$','fontsize',16)
axes(ha(4))
imageplot(DnCNNdenoised(:,:,2),[-0.1,1.1])
colorbar
set(ha(4),'position',pos{4})
title('DnCNN $\sigma_e=15/255$','fontsize',16)
