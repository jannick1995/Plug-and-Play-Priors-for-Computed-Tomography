%% Experiment 2 TVregADMM using HPC servers

% Setup the parallelization for the HPC servers. If you wish to run it on
% your own computer you most likely have to change the number of cores
parpool('local',24)
c = parcluster('local');
c.NumThreads = 2;

% Initialisations
N = 64;
N_theta = 256;
p = N;
d = p-1;

% Projection angles for overdetermined system
theta1 = linspace(0,180,N_theta + 1);
theta1 = theta1(1:end-1);

% Construct matrix and rhs for overdetermined system
[A,b,x] = paralleltomo(N,theta1,p,d);

% Construct noisy data
noise_level = 0.01;
seed = 24;
rng(seed)
e = randn(size(b));
e = noise_level * norm(b) * e / norm(e);
btilde = b + e;

% Regularization parameters
lambda_TV = logspace(-2,1,50);

XTV = zeros(N*N,3,length(lambda_TV));

RMSE_TV = zeros(3,length(lambda_TV));

% Indices for the smallest RMSE at each setup
I_TV = zeros(3,1);

optTV.display = 1;
optTV.abstol = 1e-3;
optTV.reltol = 1e-3;
optTV.maxiter = 200000;

%% Solving for overdetermined system

parfor i = 1:length(lambda_TV)
    [XTV(:,1,i),~,~,~] = TVregADMM(A,btilde,lambda_TV(i),1,optTV);
end

%% Solving for square system

rows = [];

for i = 1:4:256
    rows = [rows, (i-1)*p+1:i*p];
end

A2 = A(rows,:);
btilde2 = btilde(rows);

parfor i = 1:length(lambda_TV)
    [XTV(:,2,i),~,~,~] = TVregADMM(A2,btilde2,lambda_TV(i),1,optTV);
end

%% Solving for underdetermined system

rows = [];

for i = 1:16:256
    rows = [rows, (i-1)*p+1:i*p];
end

A3 = A(rows,:);
btilde3 = btilde(rows);

parfor i = 1:length(lambda_TV)
    [XTV(:,3,i),~,~,~] = TVregADMM(A3,btilde3,lambda_TV(i),1,optTV);
end

%% Collect the RMSE

for i = 1:3
    for j = 1:length(lambda_TV)
        RMSE_TV(i,j) = getRMSE(x,XTV(:,i,j));
    end
    
    % Store indices for min RMSE and max PSNR
    [~,I_TV(i,1)] = min(RMSE_TV(i,:));
end

clear('c')

% Save the results
save('Experiment2_TV.mat')