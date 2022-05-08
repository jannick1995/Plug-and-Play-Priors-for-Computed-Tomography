%% Experiment 1 TVregADMM using HPC servers

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

% Regularization parameters
lambda_TV = logspace(-2,2,50);

% Initializations
XTV = zeros(N*N,length(noise_level),length(lambda_TV));
RMSE_TV = zeros(length(noise_level),length(lambda_TV));

% Indices for the smallest RMSE at each noise level
I_TV = zeros(length(noise_level),1);

% Set configurations for TVregADMM
optTV.abstol = 1e-3;
optTV.reltol = 1e-3;
optTV.display = 1;
optTV.maxiter = 200000;

%% Computing the reconstructions for each noise level

for i = 1:length(noise_level)

    fprintf('Computing TVregADMM reconstructions for noise level = %.4f\n',noise_level(i))
    parfor j = 1:length(lambda_TV)
        [XTV(:,i,j),~,~,~] = TVregADMM(A,btilde(:,i),lambda_TV(j),1,optTV);
        RMSE_TV(i,j) = getRMSE(x,XTV(:,i,j));
    end
    
end

for i = 1:length(noise_level)
    % Store the best RMSE and PSNR for each noise level
    [~,I_TV(i,1)] = min(RMSE_TV(i,:));
end

save('Experiment1_TV.mat')
