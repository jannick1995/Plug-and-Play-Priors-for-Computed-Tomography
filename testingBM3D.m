%% Testing BM3D for different values of sigma 

% This script intends to inspect how the denoising capabilities of BM3D 
% depends on the denoising strength parameter sigma. This should hopefully
% give an indication of what is considered to be "large" in terms of sigma,
% and thus give an idea of what range of the regularization parameters we 
% should consider in the conducted experiments such that we avoid
% unnecessarily large denoising strength yielding poor reconstructions.

% Add path to the BM3D function
addpath 'BM3D/bm3d/';

N = 64;
X = phantom(N);
x = X(:);

% Different noise levels and denoising strengths
noise_level = [0.01, 0.03, 0.05, 0.10, 0.15];
sigma = logspace(-4,0,100);

% Initialize the arrays
Xtilde = zeros(N,N,length(noise_level));
Xtilde_s = Xtilde;
denoisedPhantoms = zeros(N,N,length(sigma),length(noise_level));
perf = zeros(length(sigma),length(noise_level));

% Construct the noisy data
seed = 34;
rng(seed)
for i = 1:length(noise_level)
    e = randn(size(x));
    e = noise_level(i) * norm(x) * e / norm(e);
    Xtilde(:,:,i) = reshape(x + e,N,N);
    % Scale Xtilde to [0,1] intensities
    lb = min(x+e);
    ub = max(x+e);
    Xtilde(:,:,i) = (Xtilde(:,:,i) - lb) / (ub - lb);
end

% Index to choose what noise level to consider 
idx = 1:5;

% Denoise the noisy phantoms 

for j = 1:length(noise_level)
    for i = 1:length(sigma)
        denoisedPhantoms(:,:,i,j) = BM3D(Xtilde(:,:,idx(j)),sigma(i));
        perf(i,j) = getRMSE(x,reshape(denoisedPhantoms(:,:,i,j),N*N,1));
    end
end

%% RMSE

semilogx(sigma,perf(:,1),'k','linewidth',1.2)
hold on
semilogx(sigma,perf(:,2),'r','linewidth',1.2)
semilogx(sigma,perf(:,3),'b','linewidth',1.2)
semilogx(sigma,perf(:,4),'r--','linewidth',1.2)
semilogx(sigma,perf(:,5),'b--','linewidth',1.2)
xlabel('$\sigma_d$','fontsize',16)
ylabel('RMSE','fontsize',16)
legend({'$||e||_2/||b||_2=0.01$','$||e||_2/||b||_2=0.03$','$||e||_2/||b||_2=0.05$','$||e||_2/||b||_2=0.10$','$||e||_2/||b||_2=0.15$'},'location','northwest','fontsize',14)
set(gca, 'XScale', 'log');
hold off

%% Individual 

idx_noise = 3;

% Index to choose what denoising strengths to consider
[~,I] = min(perf(:,idx_noise));
idx_sigma = [1, 10, 20, 40, 80, 100];

figure (1)
[ha, pos] = tight_subplot(2,3,0.04,0.03,0.03);
for i = 1:length(idx_sigma)
    axes(ha(i))
    imageplot(denoisedPhantoms(:,:,idx_sigma(i),idx_noise),[-0.1,1.1])
    if i == 3 || i == 6
        colorbar
        set(ha(i),'position',pos{i})
    end
    title(join(['$\sigma_d=$ ',num2str(sigma(idx_sigma(i)))]),'fontsize',22)
end

