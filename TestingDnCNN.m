%% Testing DnCNN for different values of sigma 

% This script intends to inspect how the denoising capabilities of DnCNN 
% depends on the trained denoising strength. This should hopefully
% give an indication of what is considered to be "large" in terms of the
% pre-trained denoising strengths.

% Add path to the DnCNN networks
addpath 'DnCNN/model/specifics/';
addpath 'DnCNN/utilities/';

N = 64;
X = phantom(N);
x = X(:);

% Different noise levels and denoising strengths (feel free to change them)
noise_level = [0.01, 0.03, 0.05, 0.10, 0.15]; % 0.10, 0.15];
sigma = 10:5:75;

% Initialize the arrays
Xtilde = zeros(N,N,length(noise_level));
denoisedPhantoms = zeros(N,N,length(sigma),length(noise_level));
residualImages = denoisedPhantoms;
perf = zeros(length(sigma),length(noise_level));

% Construct the noisy data
seed = 34;
rng(seed)
for i = 1:length(noise_level)
    e = randn(size(x));
    e = noise_level(i) * norm(x) * e / norm(e);
    Xtilde(:,:,i) = reshape(x + e,N,N);
    % Scale Xtilde to [0,1]
    lb = min(x+e);
    ub = max(x+e);
    Xtilde(:,:,i) = (Xtilde(:,:,i) - lb) / (ub - lb);
end

% Index to choose what noise level to consider 
idx = 1:length(noise_level);

% Denoise the noisy phantoms 
for j = 1:length(noise_level)
    for i = 1:length(sigma)
        % Load the correct network
        load(fullfile('model','specifics',['sigma=',num2str(sigma(i),'%02d'), ...
            '.mat']),'net');
        % Call the network
        residualImage = simplenn_matlab(net,mat2gray(Xtilde(:,:,idx(j))));
        residualImages(:,:,i,j) = residualImage(end).x;
        denoisedPhantoms(:,:,i,j) = Xtilde(:,:,idx(j)) - residualImages(:,:,i,j);
        perf(i,j) = getRMSE(x,reshape(denoisedPhantoms(:,:,i,j),N*N,1));
    end
end

%% RMSE

sigmas = sigma * (1/255);

plot(sigmas,perf(:,1),'ko-','linewidth',1.2)
hold on
plot(sigmas,perf(:,2),'ro-','linewidth',1.2)
plot(sigmas,perf(:,3),'bo-','linewidth',1.2)
plot(sigmas,perf(:,4),'ro--','linewidth',1.2)
plot(sigmas,perf(:,5),'bo--','linewidth',1.2)
xticks([10/255,20/255,30/255,40/255,50/255,60/255,70/255])
xticklabels({'10/255','20/255','30/255','40/255','50/255','60/255','70/255'})
xlabel('$\sigma_d$','fontsize',16)
ylabel('RMSE','fontsize',16)
xlim([sigmas(1),sigmas(end)])
ylim([0,0.2])
legend({'$||e||_2/||b||_2=0.01$','$||e||_2/||b||_2=0.03$','$||e||_2/||b||_2=0.05$','$||e||_2/||b||_2=0.10$','$||e||_2/||b||_2=0.15$'},'location','northwest','fontsize',14)
hold off

%% Visualize 

% Index to choose what denoising strenght to consider
idx_sigma = [1, 2, 3, 4, 10, 14]; 
idx_noise = 3;

figure (2)
[ha, pos] = tight_subplot(2,3,0.04,0.03,0.03);
for i = 1:length(idx_sigma)
    axes(ha(i))
    imageplot(denoisedPhantoms(:,:,i,idx_noise),[-0.1,1.1])
    if i == 3 || i == 6
        colorbar
        set(ha(i),'position',pos{i})
    end
    title(join(['$\sigma_d=$ ',num2str(sigma(idx_sigma(i))),'/255']),'fontsize',22)
end

