%% Code used to produce the figures used in Chapter 2

% Radon transform

P = phantom(256);
p = 0:359;
R = radon(P,p);

[ha,~] = tight_subplot(1,2,0.01,0.2,0.01);

setPlot()

axes(ha(1))
imageplot(P)
colorbar

axes(ha(2))
imagesc(R)
axis square
colorbar
set(ha(2),'YTickLabel','')
xticks([0 60 120 180 240 300 360])
xlabel('$\theta$ (degrees)','fontsize',16)


%% Back projection

P = phantom(256);
p1 = 0:179;
p2 = linspace(0,179,600);

R1 = radon(P,p1);
R2 = radon(P,p2);
[d1,d2] = size(iradon(R1,p1));
projections = [0 10 40 80 120 140 160 179];
projections2 = 0:14:179;
n = length(projections);
n2 = length(projections2);
I = zeros(d1,d2,n);
I2 = zeros(d1,d2,n2);

for i = 1:n
    r = R1(:,projections(i)+1);
    I(:,:,i) = iradon([r, r],[projections(i) projections(i)],'linear','none') / 2;
end

for i = 1:n2
    r = R1(:,projections2(i)+1);
    I2(:,:,i) = iradon([r, r],[projections2(i) projections2(i)],'linear','none') / 2;
end

I3 = zeros(d1,d2);
I4 = I3;

for i = 1:n
    I3 = I3 + I(:,:,i);
end

for i = 1:n2
    I4 = I4 + I2(:,:,i);
end

figure (1)
[ha, ~] = tight_subplot(2,2,[.01 .01],[.01 .01],[.15 .15]);
axes(ha(1));
imageplot(P)

axes(ha(2));
imageplot(I3)

axes(ha(3));
imageplot(I4)

axes(ha(4));
imageplot(iradon(R2,p2,'linear','none'))

%% FBP

P = phantom(256);
p1 = 0:179;
p2 = linspace(0,179,600);

R1 = radon(P,p1);
R2 = radon(P,p2);
[d1,d2] = size(iradon(R1,p1));
projections = [0 10 40 80 120 140 160 179];
projections2 = 0:14:179;
n = length(projections);
n2 = length(projections2);
I = zeros(d1,d2,n);
I2 = zeros(d1,d2,n2);

for i = 1:n
    r = R1(:,projections(i)+1);
    I(:,:,i) = iradon([r, r],[projections(i) projections(i)]) / 2;
end

for i = 1:n2
    r = R1(:,projections2(i)+1);
    I2(:,:,i) = iradon([r, r],[projections2(i) projections2(i)]) / 2;
end

I3 = zeros(d1,d2);
I4 = I3;

for i = 1:n
    I3 = I3 + I(:,:,i);
end

for i = 1:n2
    I4 = I4 + I2(:,:,i);
end

[ha, ~] = tight_subplot(2,2,[.01 .01],[.01 .01],[.15 .15]);

axes(ha(1));
imageplot(P)

axes(ha(2));
imageplot(I3)

axes(ha(3));
imageplot(I4)

axes(ha(4));
imageplot(iradon(R2,p2),[0,1])

%% FBP and noise

N = 256;
P = phantom(N);
theta = 0:179;
R = radon(P,theta);

noise_levels = [0.01,0.05,0.10];
Rtilde = zeros(size(R,1),size(R,2),length(noise_levels));
I = zeros(N+2,N+2,length(noise_levels));
rng(15)
for i = 1:3
    e = randn(size(R));
    e = reshape(noise_levels(i) * norm(R(:)) * e(:) / norm(e(:)),size(R,1),size(R,2));
    Rtilde(:,:,i) = R + e;
    
    I(:,:,i) = iradon(Rtilde(:,:,i),theta);
end

setPlot()
figure (1)
[ha, ~] = tight_subplot(2,3,[.01 .01],[.1 .12],[.01 .01]); 
for i = 1:length(noise_levels)
    axes(ha(i))
    imageplot(Rtilde(:,:,i),[min(R(:)),max(R(:))])
    colorbar
    axis square
    title(join(['$||e||_2/||b||_2=$ ',num2str(noise_levels(i))]),'fontsize',16)
    axes(ha(i+3))
    imageplot(I(:,:,i),[-0.1,1.1])
    colorbar
end

%% Using fbp

N = 256;
theta = 0:179;
[A,b,~] = paralleltomo(N,theta);

noise_levels = [0.01,0.05,0.10];
btilde = zeros(length(b),length(noise_levels));
fbp_sol = zeros(N,N,length(noise_levels));
rng(15)
for i = 1:3
    e = randn(size(b));
    e = noise_levels(i) * norm(b) * e / norm(e);
    btilde(:,i) = b + e;
    
    fbp_sol(:,:,i) = reshape(fbp(A,btilde(:,i),theta),N,N);
end

setPlot()
figure (1)
[ha, pos] = tight_subplot(2,3,[.01 .01],[.1 .12],[.01 .01]); 
for i = 1:length(noise_levels)
    axes(ha(i))
    imageplot(reshape(btilde(:,i),length(b)/length(theta),length(theta)),[min(b),max(b)])
    axis square
    title(join(['$||e||_2/||b||_2=$ ',num2str(noise_levels(i))]),'fontsize',16)
    axes(ha(i+3))
    imageplot(fbp_sol(:,:,i),[0,1])
end
