%% Experiment 2 PnP-BM3D using HPC servers

% Setup the parallelization for the HPC servers. If you wish to run it on
% your own computer you most likely have to change the number of cores
parpool('local',24);
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

% Regularisation parameters
lambda_BM3D = logspace(-4,1,50);

% Initializations
XBM3D = zeros(N*N,3,length(lambda_BM3D));
RMSE_BM3D = zeros(3,length(lambda_BM3D));

% Index to store the smallest RMSE
I_BM3D = zeros(3,1);

optBM3D.display = 1;
optBM3D.tol = 0;
optBM3D.maxiter = 600;

%% Solving the overdetermined system

parfor i = 1:length(lambda_BM3D)
    [XBM3D(:,1,i),~,~,~] = PnPBM3D(A,btilde,lambda_BM3D(i),1,optBM3D);
end

%% Solving the square system

rows = [];

for i = 1:4:256
    rows = [rows, (i-1)*p+1:i*p];
end

A2 = A(rows,:);
btilde2 = btilde(rows);

parfor i = 1:length(lambda_BM3D)
    [XBM3D(:,2,i),~,~,~] = PnPBM3D(A2,btilde2,lambda_BM3D(i),1,optBM3D);
end

%% Solving the underdetermined system

rows = [];

for i = 1:16:256
    rows = [rows, (i-1)*p+1:i*p];
end

A3 = A(rows,:);
btilde3 = btilde(rows);

parfor i = 1:length(lambda_BM3D)
    [XBM3D(:,3,i),~,~,~] = PnPBM3D(A3,btilde3,lambda_BM3D(i),1,optBM3D);
end

%% Collect the RMSE

for i = 1:3
    for j = 1:length(lambda_BM3D)
        RMSE_BM3D(i,j) = getRMSE(x,XBM3D(:,i,j));
    end
    
    % Store indices for min RMSE and max PSNR
    [~,I_BM3D(i,1)] = min(RMSE_BM3D(i,:));
end

% Save the results
save('Experiment2_BM3D.mat')