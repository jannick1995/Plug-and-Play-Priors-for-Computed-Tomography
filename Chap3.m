%% Code used to create figures used in Chapter 3

% Demonstrating the need of regularization in case of erroneous data
N = 64;
theta = linspace(0,179,80);
p = 64;
[A,b,x] = paralleltomo(N,theta,p);

% A has full column rank --> a unique least squares solution exists
% however, cond(A'*A) = 5.8e+10

% Construct noisy data
noise_level = 0.005;
rng(293)
e = randn(size(b));
e = noise_level * norm(b) * e / norm(e);
btilde = b + e;

% naive solution on clean and noisy data
X_naive_clean = reshape((A'*A) \ A'*b,N,N);
X_naive_noisy = reshape((A'*A) \ A'*btilde,N,N);
X = reshape(x,N,N);

setPlot()
figure (1)
[ha, pos] = tight_subplot(2,2,0.06,0.06,0.1);
axes(ha(1))
imageplot(reshape(b,length(b)/80,80),[min(btilde),max(btilde)])
axis square
colorbar;
set(ha(1),'position',pos{1})
title('$||e||_2/||b||_2=0$','fontsize',16)
axes(ha(2))
imageplot(reshape(btilde,length(b)/80,80),[min(btilde),max(btilde)])
axis square
colorbar
set(ha(2),'position',pos{2})
title('$||e||_2/||b||_2=0.005$','fontsize',16)
axes(ha(3))
imageplot(X_naive_clean,[-0.1,1.1])
colorbar
set(ha(3),'position',pos{3})
title('Naive reconstruction','fontsize',16)
axes(ha(4))
imageplot(X_naive_noisy,[-0.1,1.1])
colorbar
set(ha(4),'position',pos{4})
title('Naive reconstruction','fontsize',16)

%% Tikhonov regularized solution to the same example

lambda = 0.03;
X_tik_clean = reshape((A'*A+2*lambda*eye(size(A'*A)))\A'*b,N,N);
X_tik_noisy = reshape((A'*A+2*lambda*eye(size(A'*A)))\A'*btilde,N,N);

figure (2)
[ha, pos] = tight_subplot(2,2,0.06,0.06,0.1);
axes(ha(1))
imageplot(reshape(b,length(b)/80,80),[min(btilde),max(btilde)])
axis square
colorbar
set(ha(1),'position',pos{1})
title('$||e||_2/||b||_2=0$','fontsize',16)
axes(ha(2))
imageplot(reshape(btilde,length(b)/80,80),[min(btilde),max(btilde)])
axis square
colorbar
set(ha(2),'position',pos{2})
title('$||e||_2/||b||_2=0.005$','fontsize',16)
axes(ha(3))
imageplot(X_tik_clean,[-0.1,1.1])
colorbar
set(ha(3),'position',pos{3})
title('Tikhonov reconstruction','fontsize',16)
axes(ha(4))
imageplot(X_tik_noisy,[-0.1,1.1])
colorbar
set(ha(4),'position',pos{4})
title('Tikhonov reconstruction','fontsize',16)


%% Increasing lambda in Tikhonov regularization

lambdas = [0.1, 1, 10, 100];
Xtiks = zeros(N,N,length(lambdas));

for i = 1:length(lambdas)
    Xtiks(:,:,i) = reshape((A'*A + 2*lambdas(i) * eye(N*N)) \ A'*btilde,N,N);
end

figure (3)
[ha, pos] = tight_subplot(2,2,0.06,0.06,0.1);
for i = 1:length(lambdas)
   axes(ha(i))
   imageplot(Xtiks(:,:,i),[-0.1,1.1])
   colorbar
   set(ha(i),'position',pos{i})
   title(join(['$\lambda=$ ',num2str(lambdas(i))]),'fontsize',16)
end

%% Total Variation regularized solution to the same example

% Add TVReg to the path
addpath 'TVReg';

% TVReg solution
lambda = 0.13;
dims = [N,N];
constraint.type = 1;
constraint.c    = -inf*ones(prod(dims),1);
constraint.d    = inf*ones(prod(dims),1);
XTV_clean = reshape(tvreg_gpbb(A,b,lambda,1e-4,[N,N],constraint),N,N);
XTV_noisy = reshape(tvreg_gpbb(A,btilde,lambda,1e-4,[N,N],constraint),N,N);

figure (4)
[ha, ~] = tight_subplot(2,2,0.06,0.06,0.1);
axes(ha(1))
imageplot(reshape(b,length(b)/80,80),[min(btilde),max(btilde)])
axis square
colorbar
set(ha(1),'position',pos{1})
title('$||e||_2/||b||_2=0$','fontsize',16)
axes(ha(2))
imageplot(reshape(btilde,length(b)/80,80),[min(btilde),max(btilde)])
axis square
colorbar
set(ha(2),'position',pos{2})
title('$||e||_2/||b||_2=0.005$','fontsize',16)
axes(ha(3))
imageplot(XTV_clean,[-0.1,1.1])
colorbar
set(ha(3),'position',pos{3})
title('TV reconstruction','fontsize',16)
axes(ha(4))
imageplot(XTV_noisy,[-0.1,1.1])
colorbar
set(ha(4),'position',pos{4})
title('TV reconstruction','fontsize',16)

%% Comparison with Tikhonov

X = reshape(x,N,N);

subplot(121)
singlePlot(1:N,X(:,32),'k',1.2,16,'Pixels','Intensity',X_tik_noisy(:,32),'r')
ylim([-0.2,1.2])
legend({'Phantom','Tikhonov'},'location','north','fontsize',14)

subplot(122)
singlePlot(1:N,X(:,32),'k',1.2,16,'Pixels','Intensity',XTV_noisy(:,32),'b')
ylim([-0.2,1.2])
legend({'Phantom','TV'},'location','north','fontsize',14)

%% Varying lambda in TV

opt.k_max = 2000000;
lambdas = [0.1, 1, 10, 100];
XTVs = zeros(N,N,length(lambdas));

for i = 1:length(lambdas)
    XTVs(:,:,i) = reshape(tvreg_gpbb(A,btilde,lambdas(i),1e-4,[N,N],constraint),N,N);
end

figure (6)
[ha, pos] = tight_subplot(2,2,0.06,0.06,0.1);
for i = 1:length(lambdas)
   axes(ha(i))
   imageplot(XTVs(:,:,i),[-0.1,1.1])
   colorbar
   set(ha(i),'position',pos{i})
   title(join(['$\lambda=$ ',num2str(lambdas(i))]),'fontsize',16)
end

%% Staircasing in TV

N = 64;
X = phantomgallery('smooth',N);
x = X(:);
dims = [N,N];
tau = 1e-4*norm(x,'inf');
constraint.type = 1;
constraint.c    = -inf*ones(prod(dims),1);
constraint.d    = inf*ones(prod(dims),1);

[A,~,~] = paralleltomo(N,0:179);
b = A * x;

rng(333333)
e = randn(size(b));
e = 0.02 * norm(b) * e / norm(e);
btilde = b + e;

alpha = [0.01, 0.05, 0.1, 0.2, 0,3];
TV = zeros(N*N,length(alpha));
RMSE = zeros(length(alpha),1);

%%

for i = 1:length(alpha)
    TV(:,i) = tvreg_gpbb(A,btilde,alpha(i),tau,dims,constraint);
    RMSE(i) = getRMSE(x,TV(:,i));
end

[~,I] = min(RMSE);

TVsol = reshape(TV(:,I),N,N);

%%

figure (7)
[ha, ~] = tight_subplot(1,2,0.01,0.01,0.01); 
axes(ha(1))
imageplot(X,[-0.1,1.1])
colorbar
title('','fontsize',16)
axes(ha(2))
imageplot(TVsol,[-0.1,1.1])
colorbar
title('TV reconstruction','fontsize',16)

% figure (7)
% [ha, pos] = tight_subplot(2,2,0.06,0.06,0.1);
% axes(ha(1))
% imageplot(reshape(b,length(b)/180,180),[min(btilde),max(btilde)])
% axis square
% colorbar
% axes(ha(2))
% imageplot(reshape(btilde,length(b)/180,180),[min(btilde),max(btilde)])
% axis square
% colorbar
% axes(ha(3))
% imageplot(X,[-0.1,1.1])
% colorbar
% title('Ground truth','fontsize',16)
% axes(ha(4))
% imageplot(TVsol,[-0.1,1.1])
% colorbar
% title('TV reconstruction','fontsize',16)

%% Noisier reconstruction

N = 64;
theta = linspace(0,179,80);
p = 64;
[A,b,x] = paralleltomo(N,theta,p);

% Construct noisy data
noise_level = 0.05;
rng(293)
e = randn(size(b));
e = noise_level * norm(b) * e / norm(e);
btilde = b + e;

% Add TVReg to the path
addpath 'TVReg';

% TVReg solution
alpha = logspace(-1,1,10);
dims = [N,N];
constraint.type = 1;
constraint.c    = -inf*ones(prod(dims),1);
constraint.d    = inf*ones(prod(dims),1);
XTV = zeros(N,N,length(alpha));
RMSE = zeros(length(alpha),1);

for i = 1:length(alpha)
    XTV(:,:,i) = reshape(tvreg_gpbb(A,btilde,alpha(i),1e-4,[N,N],constraint),N,N);
    RMSE(i) = getRMSE(x,XTV(:,:,i));
end


[~,I] = min(RMSE);

%%
setPlot()
figure (1)
[ha, ~] = tight_subplot(2,2,0.06,0.06,0.1);
axes(ha(1))
imageplot(reshape(b,length(b)/80,80))
%axis square
colorbar
title('$||e||_2/||b||_2=0$','fontsize',16)
axes(ha(2))
imageplot(reshape(btilde,length(b)/80,80))
%axis square
colorbar
title('$||e||_2/||b||_2=0.005$','fontsize',16)
axes(ha(3))
imageplot(XTV(:,:,1),[0,1])
colorbar
title('Naive reconstruction','fontsize',16)
axes(ha(4))
imageplot(XTV(:,:,I),[0,1])
colorbar
title('Naive reconstruction','fontsize',16)

imageplot(XTV(:,:,I),[0,1])


