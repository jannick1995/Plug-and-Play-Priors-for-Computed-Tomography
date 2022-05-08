%% Experiment 1 PnP-DnCNN using HPC servers

% Setup the parallelization for the HPC servers. If you wish to run it on
% your own computer you most likely have to change the number of cores
parpool('local',24)
c = parcluster('local');
c.NumThreads = 2;

N = 64;
theta = 0:179;
[A,b,x] = paralleltomo(N,theta);

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
eta = logspace(-6,2,20);

% Trained noise levels
sigma_train = 10:10:75;

% Initialisations
XDnCNN_low = zeros(N*N,length(sigma_train),length(eta));
XDnCNN_med = zeros(N*N,length(sigma_train),length(eta));
XDnCNN_hi  = zeros(N*N,length(sigma_train),length(eta));

RMSE_DnCNN = zeros(length(sigma_train),length(eta),length(noise_level));

% Indices for smallest RMSE and largest PSNR
I_DnCNN_RMSE = zeros(length(noise_level),2);

% Set configurations for PnP-DnCNN
optDnCNN.tol = 0;
optDnCNN.maxiter = 600;
optDnCNN.display = 1;

%% Computing the reconstructions

for i = 1:length(sigma_train)
    fprintf('sigma = %d/%d\n',sigma_train(i),255)
    parfor j = 1:length(mu)
        % Low noise level
        [XDnCNN_low(:,i,j),~,~,~] = PnPDnCNN(A,btilde(:,1),eta(j),sigma_train(i),optDnCNN);
        % Medium noise level
        [XDnCNN_med(:,i,j),~,~,~] = PnPDnCNN(A,btilde(:,2),eta(j),sigma_train(i),optDnCNN);
        % High noise level
        [XDnCNN_hi(:,i,j),~,~,~]  = PnPDnCNN(A,btilde(:,3),eta(j),sigma_train(i),optDnCNN);
    end
    
end

% Collect RMSE and PSNR
for i = 1:length(sigma_train)
    for j = 1:length(mu)
        RMSE_DnCNN(i,j,1) = getRMSE(x,XDnCNN_low(:,i,j));
        RMSE_DnCNN(i,j,2) = getRMSE(x,XDnCNN_med(:,i,j));
        RMSE_DnCNN(i,j,3) = getRMSE(x,XDnCNN_hi(:,i,j));
    end
end

% Find smallest RMSE and largest PSNR for each noise level
for i = 1:length(noise_level)
    tmp1 = min(min(RMSE_DnCNN(:,:,i)));
    [I_DnCNN_RMSE(i,1),I_DnCNN_RMSE(i,2)] = find(RMSE_DnCNN(:,:,i) == tmp1);
end

% Remove the variable used for the parallelisation (it's not of use)
clear('c')

save('Experiment1_DnCNN.mat')