function [x, z, u, info] = PnPBM3D(A, b, lambda, rho, opt)
%
% PnPBM3D "solves" the optimization problem
%
%    (*)   arg min (1/2) * || Ax - b ||^2 + lambda * g(x),
%             x
% 
% where || . || denotes the 2-norm, and g(x) denotes the prior.
%
% The problem (*) is solved by use of ADMM, where the augmented Lagrangian
% is optimized iteratively with respect to x and z. The solution is stored
% in the vector x.
%
% Input: 
%        A      - System matrix
%
%        b      - Right hand side 
%
%        lambda - Regularization parameter
%
%        rho    - Penalty parameter 
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
% Author: Jannick Jørgensen Lønver

% Set path to the BM3D function 
addpath BM3D/bm3d;

t_start = tic;

% Global constants
DISPLAY = 0;
MAXITER = 100;
LSQRiters = 1000;
LSQRtol = 1e-6;
gamma = 1.0;
tol = 1e-3;
info.x = [];
info.z = [];
info.u = [];

% Extract dimensions
[~,n2] = size(A); % n2 = n^2

% Initialize x, z and u
x = zeros(n2,1);
z = zeros(n2,1);
u = zeros(n2,1);

if lambda < 0
    error('lambda must be nonnegative')
end

if rho < 0
   error('rho must be nonnegative') 
end
    
% If options are given, use these values (if they exist)
if nargin > 4
    if isfield(opt,'display')
        DISPLAY = opt.display;
    end
    if isfield(opt,'maxiter')
        MAXITER = opt.maxiter;
    end
    if isfield(opt,'lsqrtol')
        LSQRtol = opt.lsqrtol;
    end
    if isfield(opt,'tol')
        tol = opt.tol;
    end
    if isfield(opt,'x0')
        x = opt.x0;
    end
    if isfield(opt,'z0')
        z = opt.z0;
    end
    if isfield(opt,'u0')
        u = opt.u0;
    end
    if isfield(opt,'gamma')
        gamma = opt.gamma;
    end
end

if ~DISPLAY
    fprintf('%3s\t%10s\t%10s\t%10s\n','iter','|| x-x_old ||', ...
        '|| z-z_old ||','|| u-u_old ||');
end

for k = 1:MAXITER
    
    % Save old updates
    x_old = x;
    z_old = z;
    u_old = u;
   
    % x-update using LSQR
    [x,flag] = lsqr([A;sqrt(rho)*speye(n2)],[b;sqrt(rho)*(z_old-u_old)],LSQRtol,LSQRiters);
    
    % Check if LSQR converged within the set iterations
    if flag == 1
        fprintf('LSQR did not converge within %d iterations',LSQRiters)
    end
    
    % Store the x-iterate
    info.x = [info.x x];
    
    % Scale ztilde to [0 1]
    ztilde = x + u_old;
    lb = min(ztilde);
    ub = max(ztilde);
    ztilde_scaled = reshape((ztilde - lb) / (ub - lb),sqrt(n2),[]);
    
    % Call BM3D 
    
    % Noise intensity used for BM3D
    sigma = sqrt(lambda/rho);
    
    output_from_BM3D = BM3D(ztilde_scaled,sigma);
    
    % Rescale the denoised ztilde
    scaled_output_from_BM3D = lb + output_from_BM3D * (ub - lb);
    z = reshape(scaled_output_from_BM3D,n2,1);
    
    % Store the z-iterate
    info.z = [info.z, z];
        
    % Dual variable update
    u = u_old + (x - z);
    
    % Store the u-iterate
    info.u = [info.u, u];
    
    % Diagnostics  (normed difference of iterations)  
    info.xdiff(k) = norm(x-x_old);
    info.zdiff(k) = norm(z-z_old);
    info.udiff(k) = norm(u-u_old);
    info.Delta(k) = 1/sqrt(n2) * (info.xdiff(k) + info.zdiff(k) + info.udiff(k));
    
    if ~DISPLAY
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\n',k, ...
            info.xdiff(k), info.zdiff(k), info.udiff(k))
    end
    
    % Convergence check
    if info.Delta(k) < tol
        info.iters = k;
        break;
    end
    
    % Update rho
    rho = rho * gamma;

end

if ~DISPLAY
    toc(t_start);
end

if k >= MAXITER
    fprintf('PnP-BM3D did not converge in %d iterations\n',MAXITER)
end

end 
