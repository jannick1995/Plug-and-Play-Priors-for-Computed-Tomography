%% Experiment 1 BM3D using HPC servers

% Setup the parallelization for the HPC servers. If you wish to run it on
% your own computer you most likely have to change the number of cores
parpool('local',24)
c = parcluster('local');
c.NumThreads = 2;

N = 64;
theta = linspace(0,180,257);
theta = theta(1:end-1);
p = N;
d = p-1;
[A,b,x] = paralleltomo(N,theta,p,d);

% Relative noise levels
noise_level = [0.01, 0.13, 0.25];

e = zeros(length(b),length(noise_level));
btilde = e;
seed = 123;

% Construct noise with the corresponding relative noise levels
for i = 1:length(noise_level)
    rng(seed)
    e(:,i) = randn(size(b));
    e(:,i) = noise_level(i) * norm(b) * e(:,i) / norm(e(:,i));
    btilde(:,i) = b + e(:,i);
end

% Regularisation parameters
lambda_BM3D = logspace(-3,1,50);

% Initialisations
XBM3D = zeros(N*N,length(noise_level),length(lambda_BM3D));
RMSE_BM3D = zeros(length(noise_level),length(lambda_BM3D));

% Indices for the smallest RMSE and largest PSNR at each noise level
I_BM3D = zeros(length(noise_level),2);

% Set configurations for PnP-BM3D
optBM3D.abstol = 0;
optBM3D.maxiter = 600;
optBM3D.display = 1;

%% Computing the reconstructions for each noise level

for i = 1:length(noise_level)

    fprintf('Computing PnP-BM3D reconstructions for noise level = %.4f\n\n',noise_level(i))
    parfor j = 1:length(lambda_BM3D)
        %fprintf('Using lambda %d out of %d\n',j,length(lambda_BM3D))
        [XBM3D(:,i,j),~,~,~] = PnPBM3D(A,btilde(:,i),lambda_BM3D(j),1,optBM3D);
    end
end

for i = 1:length(noise_level)
    for j = 1:length(lambda_BM3D)
        RMSE_BM3D(i,j) = getRMSE(x,XBM3D(:,i,j));
    end
    
    % Store the best RMSE and PSNR for each noise level
    [~,I_BM3D(i,1)] = min(RMSE_BM3D(i,:));
end

clear('c')

save('Experiment1_BM3D.mat')
