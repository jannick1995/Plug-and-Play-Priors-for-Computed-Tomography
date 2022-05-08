%% Experiment 2 PnP-DnCNN using HPC servers

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

% Regularisation parameters
eta = logspace(-6,2,20);
sigma_train = 10:10:75;

%p1 = overdetermined, p2 = square, p3 = underdetermined
XDnCNN_p1 = zeros(N*N,length(sigma_train),length(eta));
XDnCNN_p2 = zeros(N*N,length(sigma_train),length(eta));
XDnCNN_p3 = zeros(N*N,length(sigma_train),length(eta));

RMSE_DnCNN = zeros(length(sigma_train),length(eta),3);

% Indices for smallest RMSE
I_DnCNN_RMSE = zeros(3,2);

% Set configurations for PnP-DnCNN
optDnCNN.tol = 0;
optDnCNN.maxiter = 600;
optDnCNN.display = 1;

%% Solving for overdetermined system

for i = 1:length(sigma_train)
    fprintf('sigma = %d/%d\n',sigma_train(i),255)
    parfor j = 1:length(mu)
        [XDnCNN_p1(:,i,j),~,~,~] = PnPDnCNN(A,btilde,eta(j),sigma_train(i),optDnCNN);
    end
end

%% Solving for square system

rows = [];

for i = 1:4:256
    rows = [rows, (i-1)*p+1:i*p];
end

A2 = A(rows,:);
btilde2 = btilde(rows);

for i = 1:length(sigma_train)
    fprintf('sigma = %d/%d\n',sigma_train(i),255)
    parfor j = 1:length(mu)
        [XDnCNN_p2(:,i,j),~,~,~] = PnPDnCNN(A2,btilde2,eta(j),sigma_train(i),optDnCNN);
    end
end

%% Solving for underdetermined system

rows = [];

for i = 1:16:256
    rows = [rows, (i-1)*p+1:i*p];
end

A3 = A(rows,:);
btilde3 = btilde(rows);

for i = 1:length(sigma_train)
    fprintf('sigma = %d/%d\n',sigma_train(i),255)
    parfor j = 1:length(mu)
        % Square 
        [XDnCNN_p3(:,i,j),~,~,~] = PnPDnCNN(A3,btilde3,eta(j),sigma_train(i),optDnCNN);
    end
end

%% Collect RMSE

for i = 1:length(sigma_train)
    for j = 1:length(mu)
        RMSE_DnCNN(i,j,1) = getRMSE(x,XDnCNN_p1(:,i,j));
        RMSE_DnCNN(i,j,2) = getRMSE(x,XDnCNN_p2(:,i,j));
        RMSE_DnCNN(i,j,3) = getRMSE(x,XDnCNN_p3(:,i,j));
    end
end

for i = 1:length(noise_level)
    % Find smallest RMSE for each setup
    tmp1 = min(min(RMSE_DnCNN(:,:,i)));
    [I_DnCNN_RMSE(i,1),I_DnCNN_RMSE(i,2)] = find(RMSE_DnCNN(:,:,i) == tmp1);
end

% Remove the variable used for the parallelisation (it's not of use)
clear('c')

% Save the results
save('Experiment2_DnCNN.mat')