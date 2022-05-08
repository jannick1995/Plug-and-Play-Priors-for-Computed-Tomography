function [x, z, u, info] = PnPDnCNN(A, b, eta, sigma, opt)
%
% PnPDnCNN "solves" the optimization problem
%
%    (*)   arg min (eta/2) * || Ax - b ||^2 + g(x),
%             x
% 
% where || . || denotes the 2-norm, and g(x) denotes the prior.
%
% The problem (*) is solved by use of ADMM, where the augmented Lagrangian
% is optimised iteratively with respect to x and z. The solution is stored
% in the vector x.
%
% Input: 
%        A      - System matrix
%
%        b      - Right hand side 
%
%        eta    - Regularization parameter
%
%        sigma  - Pre-trained denoising strength for DnCNN
%
%        opt    - struct containing the following fields:
%
%                 Displaying information at each iteration     (display)
%
%                 Tolerance                                    (tol)
%
%                 Maximum number of iterations                 (maxiter)     
%
%                 Initial z variable                           (z0)
%
%                 Initial scaled dual variable                 (u0)
%
%                 Iterations for lsqr                          (lsqriters)
%
%                 Tolerance for lsqr                           (lsqrtol)
%
% Output:
%
%        x      - Final x iteration
%
%        z      - Final z iteration
%
%        u      - Final dual variable iteration
%
%        info   - Struct containing the following attributes:
%
%                 Normed difference between x iterations        (xdiff)
%
%                 Normed difference between z iterations        (zdiff)
%
%                 Normed difference between u iterations        (udiff)
%
%                 Iterations used to obtain the solution        (iters)
%
%                 x update at each iteration                    (x)
%
%                 z update at each iteration                    (z)
%
%                 u update at each iteration                    (u)
%
% Author: Jannick Jørgensen Lønver

% Set path to DnCNN
addpath 'DnCNN/utilities';
addpath 'DnCNN/model';
addpath 'DnCNN/model/specifics';

% Check if sigma is a proper noise level
if sum(sigma == 10:5:75) ~= 1
    error('sigma must be in [10,75] with increments of 5');
else
    modelSigma  = min(75,max(10,round(sigma/5)*5));
    load(fullfile('model','specifics',['sigma=',num2str(modelSigma,'%02d'), ...
    '.mat']),'net');
end

if eta < 0
    error('eta must be nonnegative');
end

t_start = tic;

% Global constants
DISPLAY = 0;
MAXITER = 100;
tol = 1e-3;
LSQRiters = 1000;
LSQRtol = 1e-6;

% Extract dimensions
[~,n2] = size(A); % n2 = n^2

% Initialize x, z and u
x = zeros(n2,1);
z = zeros(n2,1);
u = zeros(n2,1);
   
% If options are given, use these values (if they exist)
if nargin > 4
    if isfield(opt,'display')
        DISPLAY = opt.display;
    end
    if isfield(opt,'maxiter')
        MAXITER = opt.maxiter;
    end
    if isfield(opt,'tol')
        tol = opt.tol;
    end
    if isfield(opt,'z0')
        z = opt.z0;
    end
    if isfield(opt,'u0')
        u = opt.u0;
    end
    if isfield(opt,'lsqriters')
        LSQRiters = opt.lsqriters;
    end
    if isfield(opt,'lsqrtol')
        LSQRtol = opt.lsqrtol; 
    end
end

if ~DISPLAY
    fprintf('%3s\t%10s\t%10s\t%10s\n','iter','|| x-x_old ||', ...
        '|| z-z_old ||','|| u-u_old ||');
end

info.x = [];
info.z = [];
info.u = [];
info.conv = 0;

for k = 1:MAXITER
    
    % Save old updates
    x_old = x;
    z_old = z;
    u_old = u;
   
    % x-update using LSQR (do not form the matrices explicitly)
    [x,flag] = lsqr([sqrt(eta)*A;1/sigma*speye(n2)],[sqrt(eta)*b;1/sigma*(z-u)],LSQRtol,LSQRiters);
    
    % Check if LSQR converged in set iterations
    if flag == 1
        fprintf('LSQR did not converge a in %d iterations\n',LSQRiters)
    end
    
    info.x = [info.x, x];
    
    % z-update using a call to DnCNN
    ztilde = x + u_old;
    lb = min(ztilde);
    ub = max(ztilde);
    
    % Scale ztilde to [0 1]
    ztilde_scaled = reshape((ztilde - lb) / (ub - lb),sqrt(n2),sqrt(n2));
    
    % Use DnCNN to obtain the residual mapping
    residual_mapping = simplenn_matlab(net,ztilde_scaled);
    
    % Denoise ztilde
    output_from_denoiser = ztilde_scaled - residual_mapping(end).x;
    
    % Rescale 
    scaled_output_from_denoiser = lb + output_from_denoiser * (ub - lb);
    z = double(reshape(scaled_output_from_denoiser,n2,1));
    info.z = [info.z, z];
        
    % Dual-variable update
    u = u_old + (x - z);
    info.u = [info.u, u];
    
    % Diagnostics    
    info.xdiff(k) = norm(x-x_old);
    info.zdiff(k) = norm(z-z_old);
    info.udiff(k) = norm(u-u_old);
    
    if ~DISPLAY
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\n',k, ...
            info.xdiff(k), info.zdiff(k), info.udiff(k))
    end
    
    % Stopping criteria
    if 1/sqrt(n2) * (info.xdiff(k) + info.zdiff(k) + info.udiff(k)) < tol
        info.iters = k;
        break;
    end

end

if k >= MAXITER
    fprintf('PnP-DnCNN did not converge within %d iterations\n',MAXITER)
end

if ~DISPLAY
    toc(t_start);
end

end 
