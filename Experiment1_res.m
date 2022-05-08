%% Experiment 1 using 256 projections (similar geometry)

% Add path to the results
addpath 'Experiments';

% Load the results obtained
load('Experiment1_TV.mat')
load('Experiment1_BM3D.mat')
load('Experiment1_DnCNN.mat')

% Convert the arrays containing PnP-DnCNN into one
XDnCNN = zeros(N*N,length(sigma_train),length(eta),length(noise_level));
XDnCNN(:,:,:,1) = XDnCNN_low;
XDnCNN(:,:,:,2) = XDnCNN_med;
XDnCNN(:,:,:,3) = XDnCNN_hi;

% Construct simple lsqr and fbp solutions to compare
Xfbp = zeros(N*N,3);
RMSE_FBP = zeros(3,1);

for i = 1:3
   Xfbp(:,i) = fbp(A,btilde(:,i),theta);
   RMSE_FBP(i) = getRMSE(x,Xfbp(:,i));
end

%% RMSE plots

figure (1)
hold on
for i = 1:length(noise_level)
    semilogx(eta,RMSE_DnCNN(1,:,i),'linewidth',1.2)
end
xlabel('$\eta$','fontsize',16)
ylabel('RMSE','fontsize',16)
xlim([eta(1),eta(13)])
%ylim([0,0.1])
legend({'Noise level = 0.01','Noise level = 0.13','Noise level = 0.25'},'fontsize',14)
title('PnP-DnCNN','fontsize',16)
hold off

figure (2)
hold on
for i = 1:length(noise_level)
    semilogx(lambda_BM3D,RMSE_BM3D(i,:),'linewidth',1.2)
end
xlabel('$\lambda$','fontsize',16)
ylabel('RMSE','fontsize',16)
xlim([lambda_BM3D(1),lambda_BM3D(20)])
legend({'Noise level = 0.01','Noise level = 0.13','Noise level = 0.25'},'fontsize',14)
title('PnP-BM3D','fontsize',16)
hold off

figure (3)
hold on
for i = 1:length(noise_level)
    semilogx(lambda_TV,RMSE_TV(i,:),'linewidth',1.2)
end
xlabel('$\lambda$','fontsize',16)
ylabel('RMSE','fontsize',16)
legend({'Noise level = 0.01','Noise level = 0.13','Noise level = 0.25'},'fontsize',14)
title('TV','fontsize',16)
hold off

%% Reconstructions based on RMSE

for i = length(noise_level):-1:1
    figure (i)
    [ha, pos] = tight_subplot(2,2,0.06,0.06,0.1);
    
    axes(ha(1))
    imageplot(reshape(Xfbp(:,i),N,N),[-0.1,1.1])
    colorbar
    set(ha(1),'position',pos{1})
    title(sprintf('FBP RMSE = %.4f',RMSE_FBP(i)),'fontsize',16)
    
    axes(ha(2))
    imageplot(reshape(XDnCNN(:,I_DnCNN_RMSE(i,1),I_DnCNN_RMSE(i,2),i),N,N),[-0.1,1.1])
    colorbar
    set(ha(2),'position',pos{2})
    title(sprintf('PnP-DnCNN RMSE = %.4f',RMSE_DnCNN(I_DnCNN_RMSE(i,1),I_DnCNN_RMSE(i,2),i)),'fontsize',16)
    
    axes(ha(3))
    imageplot(reshape(XBM3D(:,i,I_BM3D(i,1)),N,N),[-0.1,1.1])
    colorbar
    set(ha(3),'position',pos{3})
    title(sprintf('PnP-BM3D RMSE = %.4f',RMSE_BM3D(i,I_BM3D(i,1))),'fontsize',16)
    
    axes(ha(4))
    imageplot(reshape(XTV(:,i,I_TV(i,1)),N,N),[-0.1,1.1])
    colorbar
    set(ha(4),'position',pos{4})
    title(sprintf('TVregADMM RMSE = %.4f',RMSE_TV(i,I_TV(i,1))),'fontsize',16) 
    
end

%%

diffs = zeros(N*N,length(noise_level),3);
lb = zeros(3,1);
ub = zeros(3,1);

for i = 1:length(noise_level)
    diffs(:,i,1) = abs(x - XDnCNN(:,I_DnCNN_RMSE(i,1),I_DnCNN_RMSE(i,2),i));
    diffs(:,i,2) = abs(x - XBM3D(:,i,I_BM3D(i,1)));
    diffs(:,i,3) = abs(x - XTV(:,i,I_TV(i,1)));
end

for i = 1:length(noise_level)
    figure (i)
    [ha, ~] = tight_subplot(1,3,0.06,0.06,0.01);
    axes(ha(1))
    imageplot(reshape(diffs(:,i,1),N,N),[min(diffs(:,i,1)),max(diffs(:,i,1))])
    colorbar
    title('PnP-DnCNN','fontsize',16)
    axes(ha(2))
    imageplot(reshape(diffs(:,i,2),N,N),[min(diffs(:,i,1)),max(diffs(:,i,1))])
    colorbar
    title('PnP-BM3D','fontsize',16)
    axes(ha(3))
    imageplot(reshape(diffs(:,i,3),N,N),[min(diffs(:,i,1)),max(diffs(:,i,1))])
    colorbar
    title('TV','fontsize',16)
end

%% Comparing slices of the phantom

% Indicates what noise level for which the best solution will be used
idx_noise = 1;

X = reshape(x,N,N);
XDnCNN_best = reshape(XDnCNN(:,I_DnCNN_RMSE(idx_noise,1),I_DnCNN_RMSE(idx_noise,2),idx_noise),N,N);
XBM3D_best = reshape(XBM3D(:,idx_noise,I_BM3D(idx_noise,1)),N,N);
XTV_best = reshape(XTV(:,idx_noise,I_TV(idx_noise,1)),N,N);

figure (1)
subplot(131)
singlePlot(1:N,X(:,32),'k',1.2,16,'Pixels','Intensity',XDnCNN_best(:,32),'b')
legend({'Phantom','PnP-DnCNN'},'location','north','fontsize',14)
ylim([-0.2,1.2])

subplot(132)
singlePlot(1:N,X(:,32),'k',1.2,16,'Pixels','Intensity',XBM3D_best(:,32),'b')
legend({'Phantom','PnP-BM3D'},'location','north','fontsize',14)
ylim([-0.2,1.2])

subplot(133)
singlePlot(1:N,X(:,32),'k',1.2,16,'Pixels','Intensity',XTV_best(:,32),'b')
legend({'Phantom','TV'},'location','north','fontsize',14)
ylim([-0.2,1.2])
