%% Comparing TVReg to TVregADMM
%
% Test difference of TVReg and TVregADMM's reconstructions for a fixed
% value of lambda (low, medium, and high regularisation), as we increase 
% the iterations.

% Add TVReg and TVregADMM to the path
addpath 'TVReg';
addpath 'TVregADMM';

% Setup the experiment
N = 64; % number of pixels in each direction
theta = 0:179; % projection angles
[A,b,x] = paralleltomo(N,theta); 

% Configurations needed for TVReg
dims = [N,N]; % image size
tau = 1e-4*norm(x,'inf'); % smoothing parameter

% While TVReg can solve the TV problem using non-negativity constraints,
% which is the more correct option, as attenuations can not be negative. We
% solve it without the non-negativity constraint, since this is not 
% incorporated in the TVregADMM algorithm.
constraint.type = 1;
constraint.c    = -inf*ones(prod(dims),1);
constraint.d    = inf*ones(prod(dims),1);

% Initialisations
iters = [50 100 200 400 800 1600];
lambda = [0.1, 1, 10]; 
rho = 1; % can be changed to increase convergence rate
XTVregADMM = zeros(N*N,length(iters),length(lambda));
XTVReg = zeros(N*N,length(iters),length(lambda));
sol_diff = zeros(N*N,length(iters),length(lambda));
norm_diff = zeros(length(iters),length(lambda));
objvals = zeros(2,length(iters),length(lambda));

% Setting the algorithms to run until max iterations are reached
optTVregADMM.abstol = 0;
optTVregADMM.reltol = 0;
optTVregADMM.display = 1;
optTVReg.epsb_rel = 0;

% Experiment
for i = 1:length(iters)
    fprintf('Using %d iterations (%d out %d)\n',iters(i),i,length(iters))
    optTVregADMM.maxiter = iters(i);
    optTVReg.k_max = iters(i);
    
    for j = 1:length(lambda)
        % TVregADMM solution
        [XTVregADMM(:,i,j),~,~,info] = TVregADMM(A,b,lambda(j),rho,optTVregADMM);
        objvals(1,i,j) = info.objval(end);
        
        % TVReg solution
        [XTVReg(:,i,j),fx,~,~,~,~] = tvreg_gpbb(A,b,lambda(j),tau,dims,constraint,optTVReg);
        objvals(2,i,j) = fx;
        
        % Absolute reconstruction difference
        sol_diff(:,i,j) = abs(XTVregADMM(:,i,j) - XTVReg(:,i,j));
        
        % Normed reconstruction difference
        norm_diff(i,j) = norm(XTVregADMM(:,i,j) - XTVReg(:,i,j));
    end
    
end

save('TVRegVSTVregADMM.mat')

%% Visualise

% Load the saved results
load('TVRegVsTVregADMM.mat')

setPlot()

idx = 3; % index for what lambda to consider
reconIdx = [4 5 6]; % choose what reconstructions to see (iterations)

figure (1)
semilogy(iters,norm_diff(:,1),'bo-','linewidth',1.2)
hold on 
semilogy(iters,norm_diff(:,2),'ro-','linewidth',1.2)
semilogy(iters,norm_diff(:,3),'ko-','linewidth',1.2)
xlabel('Iterations','fontsize',16)
ylabel('$||x_{TVregADMM}-x_{TVReg}||_2$','fontsize',16)
legend({'$\lambda=0.1$','$\lambda=1$','$\lambda=10$'},'location','northeast','fontsize',14)

figure (2)
semilogy(iters,objvals(1,:,1),'ro-','linewidth',1.2)
hold on 
semilogy(iters,objvals(2,:,1),'bo-','linewidth',1.2)
xlabel('Iterations','fontsize',16)
ylabel('Objective value','fontsize',16)
legend({'TVregADMM','TVReg'},'location','northeast','fontsize',14)
hold off
ax=axes;
set(ax,'units','normalized','position',[0.58,0.3,0.3,0.3])
box(ax,'on')
semilogy(iters,objvals(1,:,1),'ro-','linewidth',1.2)
hold on
semilogy(iters,objvals(2,:,1),'bo-','linewidth',1.2)
set(ax,'xlim',[0,500],'ylim',[34,34.5],'YScale','log')

% figure (3)
% [ha,~] = tight_subplot(2,3,0.01,0.03,0.03);
% for i = 1:length(iters)
%     axes(ha(i));
%     imageplot(reshape(sol_diff(:,i,idx),N,N))
%     colorbar
%     title(sprintf('%d iterations',iters(i)),'fontsize',16)
% end

figure (4)
%[ha, pos] = tight_subplot(2,3,0.01,0.03,0.03);
[ha, pos] = tight_subplot(2,3,0.01,0.03,0.03);
for i = 1:length(reconIdx)
    axes(ha(i));
    imageplot(reshape(XTVregADMM(:,reconIdx(i),idx),N,N),[0,1])
    colorbar
    title(sprintf('%d iterations',iters(reconIdx(i))),'fontsize',16)
    axes(ha(i+3));
    imageplot(reshape(XTVReg(:,reconIdx(i),idx),N,N),[0,1])
    colorbar
    %title(sprintf('%d iterations',iters(reconIdx(i))),'fontsize',16)
end

%% Comparisons with noise as well

addpath 'TVReg';
addpath 'TVregADMM';

% Setup the experiment (if not already saved in memory)
N = 64; % number of pixels in each direction
theta = 0:179; % projection angles
[A,b,x] = paralleltomo(N,theta);

% Construct noisy rhs
seed = 53;
rng(seed)
noise_level = 0.02; % feel free to change it
e = randn(size(b));
e = noise_level * norm(b) * e / norm(e);
btilde = b + e;

% Configurations needed for TVReg
dims = [N,N]; % image size
tau = 1e-4*norm(x,'inf'); % smoothing parameter

% Set constraints for TVReg
constraint.type = 1;
constraint.c    = -inf*ones(prod(dims),1);
constraint.d    = inf*ones(prod(dims),1);

% Initialisations
iters = [50 100 200 400 800 1600];
lambda = [0.1, 1, 10]; 
rho = 1; % can be changed to increase convergence rate
XTVregADMM_n = zeros(N*N,length(iters),length(lambda));
XTVReg_n = zeros(N*N,length(iters),length(lambda));
sol_diff_n = zeros(N*N,length(iters),length(lambda));
norm_diff_n = zeros(length(iters),length(lambda));
objvals_n = zeros(2,length(iters),length(lambda));

% Setting the algorithms to run until max iterations are reached
optTVregADMM.abstol = 0;
optTVregADMM.reltol = 0;
optTVregADMM.display = 1;
optTVReg.epsb_rel = 0;

% Experiment
for i = 1:length(iters)
    fprintf('Using %d iterations (%d out %d)\n',iters(i),i,length(iters))
    optTVregADMM.maxiter = iters(i);
    optTVReg.k_max = iters(i);
    
    for j = 1:length(lambda)
        % TVregADMM solution
        [XTVregADMM_n(:,i,j),~,~,info_n] = TVregADMM(A,btilde,lambda(j),rho,optTVregADMM);
        objvals_n(1,i,j) = info_n.objval(end);
        
        % TVReg solution
        [XTVReg_n(:,i,j),fx_n,~,~,~,~] = tvreg_gpbb(A,btilde,lambda(j),tau,dims,constraint,optTVReg);
        objvals_n(2,i,j) = fx_n;
        
        % Absolute reconstruction difference
        sol_diff_n(:,i,j) = abs(XTVregADMM_n(:,i,j) - XTVReg_n(:,i,j));
        
        % Normed reconstruction difference
        norm_diff_n(i,j) = norm(XTVregADMM_n(:,i,j) - XTVReg_n(:,i,j));
    end
    
end

save('TVRegVSTVregADMM_noise.mat')

%% Results

load('TVRegVSTVregADMM_noise.mat')


% Fix the TVregADMM solution, and compare with TVReg through the iterations
norm_rel_n = zeros(length(iters),length(lambda));

for i = 1:length(iters)
    for j = 1:length(lambda)
       norm_rel_n(i,j) = norm(XTVregADMM_n(:,3,j) - XTVReg_n(:,i,j)); 
    end
end

setPlot()

idx = 3; % index for what lambda to consider
reconIdx = [1 2 3]; % choose what reconstructions to see (iterations)

figure (6)
semilogy(iters,norm_diff_n(:,1),'bo-','linewidth',1.2)
hold on 
semilogy(iters,norm_diff_n(:,2),'ro-','linewidth',1.2)
semilogy(iters,norm_diff_n(:,3),'ko-','linewidth',1.2)
xlabel('Iterations','fontsize',16)
ylabel('$||x_{TVregADMM}-x_{TVReg}||_2$','fontsize',16)
legend({'$\lambda=0.1$','$\lambda=1$','$\lambda=10$'},'location','northeast','fontsize',14)

figure (7)
semilogy(iters,norm_rel_n(:,1),'bo-','linewidth',1.2)
hold on 
semilogy(iters,norm_rel_n(:,2),'ro-','linewidth',1.2)
semilogy(iters,norm_rel_n(:,3),'ko-','linewidth',1.2)
xlabel('Iterations','fontsize',16)
ylabel('$||x_{TVregADMM}-x_{TVReg}||_2$','fontsize',16)
legend({'$\lambda=0.1$','$\lambda=1$','$\lambda=10$'},'location','northeast','fontsize',14)

figure (8)
[ha, ~] = tight_subplot(1,2,[.1 .07],[.1 .05],[.07 .01]);
axes(ha(1));
singlePlot(iters,objvals_n(1,:,1),'ro-',1.2,16,'Iterations','$f(x)$',objvals_n(2,:,1),'bo-')
legend({'TVregADMM','TVReg'},'location','northeast','fontsize',14)
axes(ha(2));
singlePlot(iters,objvals_n(1,:,1),'ro-',1.2,16,'Iterations','$f(x)$',objvals_n(2,:,1),'bo-')
legend({'TVregADMM','TVReg'},'location','northeast','fontsize',14)
ylim([30,50])

figure (9)
[ha,~] = tight_subplot(2,3,0.01,0.03,0.03);
for i = 1:length(iters)
    axes(ha(i));
    imageplot(reshape(sol_diff_n(:,i,idx),N,N))
    title(sprintf('%d iterations',iters(i)),'fontsize',16)
end

figure (10)
[ha, ~] = tight_subplot(2,3,0.01,0.03,0.03);
for i = 1:length(reconIdx)
    axes(ha(i));
    imageplot(reshape(XTVregADMM_n(:,reconIdx(i),idx),N,N),[0,1])
    title(sprintf('%d iterations',iters(reconIdx(i))),'fontsize',16)
    axes(ha(i+3));
    imageplot(reshape(XTVReg_n(:,reconIdx(i),idx),N,N),[0,1])
    title(sprintf('%d iterations',iters(reconIdx(i))),'fontsize',16)
end

%% Testing the imact of the primal and dual tolerance

% Add TVregADMM to the path
addpath 'TVregADMM';

% Setup the experiment
N = 64; % number of pixels in each direction
theta = 0:179; % projection angles
[A,b,x] = paralleltomo(N,theta); 

seed = 54;
rng(seed)
noise_level = 0.03;
e = randn(size(b));
e = noise_level * norm(b) * e / norm(e);
btilde = b + e;

% Initialisations
tols = [1e-2, 1e-3, 1e-4, 1e-5];
%lambda = [0.1, 1, 10];
lambda = 1;
rho = 1; % can be changed to increase convergence rate
XTVregADMM_tol = zeros(N*N,length(tols),length(lambda));
objvals_tol = zeros(length(tols),length(lambda));
rel_error_tol = zeros(length(tols),length(lambda));

% Setting for TVregADMM
optTVregADMM.maxiter = 100000;

for i = 1:length(tols)
    fprintf('Using tol = %.6f (%d out of %d)\n\n',tols(i),i,length(tols))
    optTVregADMM.abstol = tols(i);
    optTVregADMM.reltol = tols(i);
    for j = 1:length(lambda)
        [XTVregADMM_tol(:,i,j),~,~,info_tol] = TVregADMM(A,btilde,lambda(j),rho,optTVregADMM);
        rel_error_tol(i,j) = norm(x - XTVregADMM_tol(:,i,j)) / norm(x);
        objvals_tol(i,j) = info_tol.objval(end);
    end
end

% It took 5241.070496 seconds to solve TV for lambda = 1 with tol = 1e-5

% It took 163 seconds for lambda = 1 with tol = 1e-4, so not that bad

% TVregADMMtoltestNoise:
% 1e-2: 17   iterations --> 3.327   seconds
% 1e-3: 135  iterations --> 24.164  seconds
% 1e-4: 653  iterations --> 131.169 seconds
% 1e-5: 2519 iterations --> 718.613 seconds

save('TVregADMMtolTestNoise.mat')

%%

%load('TVregADMMtolTest.mat')
load('TVregADMMtolTestNoise.mat')

%% Visualize

figure (1)
tols_string = ["$10^{-2}$","$10^{-3}$","$10^{-4}$","$10^{-5}$"];

subplot(121)
semilogx(tols,objvals_tol,'ko-','linewidth',1.2)
xlabel('$\epsilon^{abs}=\epsilon^{rel}$','fontsize',16)
ylabel('$\frac{1}{2}||Ax-b||_2^2+\sum_{i}||D_ix||_2$','fontsize',16)

subplot(122)
semilogx(tols,rel_error_tol,'ko-','linewidth',1.2)
xlabel('$\epsilon^{abs}=\epsilon^{rel}$','fontsize',16)
ylabel('$||x-\hat{x}||_2/||x||_2$','fontsize',16)

figure (2)
[ha,pos] = tight_subplot(2,2,0.06,0.06,0.1); 
for i = 1:length(tols)
    axes(ha(i))
    imageplot(reshape(XTVregADMM_tol(:,i),N,N),[-0.1,1.1])
    colorbar
    set(ha(i),'position',pos{i})
    title(join(['$\epsilon^{abs}=\epsilon^{rel}=$ ',tols_string(i)]),'fontsize',16)
end

