%% Results to experiment 2

addpath 'Experiments'

% Load saved results from HPC
load('Experiment2_TV.mat')
load('Experiment2_BM3D.mat')
load('Experiment2_DnCNN.mat')

proj = [256, 64, 16];

% Construct FBP for each configuration
Xfbp = zeros(N*N,length(proj));
Xfbp(:,1) = fbp(A,btilde,theta1);
Xfbp(:,2) = fbp(A2,btilde2,theta1(1:4:end));
Xfbp(:,3) = fbp(A3,btilde3,theta1(1:16:end));
RMSE_FBP = zeros(length(proj),1);

for i = 1:length(proj)
    RMSE_FBP(i) = getRMSE(x,Xfbp(:,i));
end

%% RMSE plots

cols = ['k','r','b'];
idx_DnCNN = 1;

figure (1)
hold on
for i = 1:length(proj)
    semilogx(eta,RMSE_DnCNN(idx_DnCNN,:,i),cols(i),'linewidth',1.2)
end
hold off
xlabel('$\eta$','fontsize',16)
ylabel('RMSE','fontsize',16)
xlim([eta(1),eta(11)])
legend({'256 projections','64 projections','16 projections'},'fontsize',14)
title(join(['PnP-DnCNN ','$\sigma_d=$ ',num2str(sigma_train(idx_DnCNN)),'/255']),'fontsize',16)

%ax=axes;
%set(ax,'units','normalized','position',[0.58,0.6,0.3,0.3])
%box(ax,'on')
% hold on
% for i = 1:3
%     semilogx(mu,RMSE_DnCNN(idx_DnCNN,:,i),cols(i),'linewidth',1.2)
% end
%set(ax,'xlim',[0,0.05],'ylim',[0,0.1])
%hold off

figure (2)
hold on
for i = 1:length(proj)
    semilogx(lambda_BM3D,RMSE_BM3D(i,:),cols(i),'linewidth',1.2)
end
xlabel('$\lambda$','fontsize',16)
ylabel('RMSE','fontsize',16)
xlim([lambda_BM3D(1),lambda_BM3D(21)])
legend({'256 projections','64 projections','16 projections'},'fontsize',14)
title('PnP-BM3D','fontsize',16)
hold off

%ax=axes;
%set(ax,'units','normalized','position',[0.58,0.6,0.3,0.3])
%box(ax,'on')
% hold on
% for i = 1:3
%     semilogx(mu,RMSE_DnCNN(idx_DnCNN,:,i),cols(i),'linewidth',1.2)
% end
% %set(ax,'xlim',[0,0.1],'ylim',[0,0.1])
% hold off

figure (3)
hold on
for i = 1:length(proj)
    semilogx(lambda_TV,RMSE_TV(i,:),cols(i),'linewidth',1.2)
end
xlabel('$\lambda$','fontsize',16)
ylabel('RMSE','fontsize',16)
legend({'256 projections','64 projections','16 projections'},'location','northwest','fontsize',14)
title('TV','fontsize',16)
hold off

%% Reconstructions using RMSE

% For each noise level, a 2-by-2 subplot is constructed containing the
% fbp solution, and the solutions obtained from PnP and TV with the
% smallest RMSE.

for i = 3:-1:1
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


%% Differences in the reconstructions

diffs = zeros(N*N,length(proj),3);

for i = 1:length(proj)
    diffs(:,i,1) = abs(x - XDnCNN(:,I_DnCNN_RMSE(i,1),I_DnCNN_RMSE(i,2),i));
    diffs(:,i,2) = abs(x - XBM3D(:,i,I_BM3D(i,1)));
    diffs(:,i,3) = abs(x - XTV(:,i,I_TV(i,1)));
end

for i = 1:length(proj)
    figure (i)
    [ha, ~] = tight_subplot(1,3,0.06,0.06,0.01);
    axes(ha(1))
    imageplot(reshape(diffs(:,i,1),N,N))
    %title(join(['PnP-DnCNN ','$||e||/||e||=$',num2str(noise_level(i))]),'fontsize',16)
    title('PnP-DnCNN','fontsize',16)
    axes(ha(2))
    imageplot(reshape(diffs(:,i,2),N,N))
    %title(join(['PnP-BM3D ','$||e||/||e||=$',num2str(noise_level(i))]),'fontsize',16)
    title('PnP-BM3D','fontsize',16)
    axes(ha(3))
    imageplot(reshape(diffs(:,i,3),N,N))
    %title(join(['TV ','$||e||/||e||=$',num2str(noise_level(i))]),'fontsize',16)
    title('TV','fontsize',16)
end
