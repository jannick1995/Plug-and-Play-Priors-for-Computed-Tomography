function [x, z, u, info] = TVregADMM(A, b, lambda, rho, opt)
%
% TVregADMM solves the Total Variation optimization problem
%
%    (*)   arg min (1/2) * || Ax - b ||^2 + lambda * TV(x),
%             x
% 
% where || . || denotes the 2-norm and
%
%       TV(x) = sum_{i=1}^{n^2} || D_i x ||, i = 1,...,n^2.
%
% D_i x denotes the 2-by-1 vector containing the horizontal and
% vertical first order finite difference of x at pixel i.
%
% The problem (*) is solved by use of ADMM, where the augmented Lagrangian
% is optimised iteratively with respect to x and z. The solution is stored
% in the vector x.
%
% Input: 
%
%        A      - System matrix (m by n)
%
%        b      - Right hand side (m by 1)
%
%        lambda - Regularization parameter (non-negative)
%
%        rho    - penalty parameter (non-negative)
%
%        opt    - struct containing the following fields:
%
%                 Displaying information at each iteration     (display)
%
%                 Absolute tolerance                           (abstol)
%
%                 Relative tolerance                           (reltol)
%
%                 Maximum number of iterations                 (maxiter)     
%
%                 Initial z variable (2*n2-by-1)               (z0)
%
%                 Initial scaled dual variable (2*n2-by-1)     (u0)
%
%                 Iterations for lsqr                          (lsqriters)
%
%                 Tolerance for lsqr                           (lsqrtol)
%
% Output:
%
%        x      - Solution from TVregADMM
%
%        z      - Gradient of solution from TVregADMM
%
%        u      - Dual variable
%
%        info   - Struct containing the following fields:
%
%                 Objective value at iteration k               (objval)
%
%                 Norm of the primal residual at iteration k   (r_norm)
%
%                 Norm of the dual residual at iteration k     (s_norm)
%
%                 Tolerance for primal residual at iteration k (eps_pri)
%
%                 Tolerance for dual residual at iteration k   (eps_dual)
%
%                 Iterations used to obtain the solution       (iters)
%
%                 x update at each iteration                    (x)
%
%                 z update at each iteration                    (z)
%
%                 u update at each iteration                    (u)
%
% Author: Jannick Jørgensen Lønver

t_start = tic;

% Default constants
DISPLAY = 0;
MAXITER = 1000;
ABSTOL = 1e-2;
RELTOL = 1e-2;
LSQRiters = 1000;
LSQRtol = 1e-6;

% Extract total number of pixels n^2
[~,n2] = size(A);

% Default initialisations
z = zeros(2*n2,1);
u = zeros(2*n2,1);

% Check if input is valid
if lambda < 0
    error('lambda has to be nonnegative');
end
if rho < 0
    error('rho has to be nonnegative');
end

% If options are given, use these values (if they exist)
if nargin > 4
    if isfield(opt,'display')
        DISPLAY = opt.display;
    end
    if isfield(opt,'abstol')
        ABSTOL = opt.abstol;
    end
    if isfield(opt,'reltol')
        RELTOL = opt.reltol;
    end
    if isfield(opt,'maxiter')
        MAXITER = opt.maxiter;
    end
    if isfield(opt,'lsqriters')
        LSQRiters = opt.lsqriters;
    end
    if isfield(opt,'lsqrtol')
        LSQRtol = opt.lsqrtol;
    end
    if isfield(opt,'z0')
        z = opt.z0;
    end
    if isfield(opt,'u0')
        u = opt.u0;
    end
end

if ~DISPLAY
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n','iter','|| r ||', ...
        'eps pri','|| s ||', 'eps dual','objective');
end

info.x = [];
info.z = [];
info.u = [];

% Construct the global finite difference matrix D
D = finiteDiffMatrix(sqrt(n2));

for k = 1:MAXITER
    
    % x-update using LSQR
    [x,flag] = lsqr([A;sqrt(rho)*D],[b;sqrt(rho)*(z+u)],LSQRtol,LSQRiters);
    
    % Check if LSQR converged within the set number of iterations
    if flag == 1
        fprintf('LSQR did not converge in %d iterations',LSQRiters)
    end
    
    info.x = [info.x, x];
    
    % Store the matrix vector product D*x
    Dx = D*x;
    
    % Store previous z before updating it
    z_old = z;
    
    % z-update using shrinkage
    for i = 1:2:length(z)
        z(i:i+1) = shrinkage(Dx(i:i+1),lambda,rho,u(i:i+1));
    end 
    
    % z = shrinkage(Dx(i,:),lambda,rho,u(i,:))
    
    info.z = [info.z, z];
    
    % Dual variable update
    u = u + (z - Dx);
    info.u = [info.u, u];
    
    % Diagnostics
    info.objval(k) = objective(A, b, lambda, x, z);
    
    info.r_norm(k) = norm(z - Dx);
    info.s_norm(k) = norm(rho * D'*(z_old - z));
    
    info.eps_pri(k)  = sqrt(2*n2) * ABSTOL + RELTOL * max(norm(Dx),norm(z));
    info.eps_dual(k) = sqrt(n2) * ABSTOL + RELTOL * norm(rho*D'*u);
    
    if ~DISPLAY
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n',k, ...
            info.r_norm(k), info.eps_pri(k), info.s_norm(k), ...
            info.eps_dual(k), info.objval(k))
    end
    
    % Convergence check
    if (info.r_norm(k) < info.eps_pri(k) && ...
            info.s_norm(k) < info.eps_dual(k))
        info.iters = k;
        break;
    end
   
end

if k >= MAXITER
    fprintf('TVregADMM did not converge within %d iterations\n',MAXITER)
end

if ~DISPLAY
    toc(t_start);
end

end 

function p = objective(A, b, lambda, x, z)

tmp = 0;

for i = 1:2:length(z)
    tmp = tmp + norm(z(i:i+1),2);
end
    
p = (1/2) * norm(A*x - b)^2 + lambda * tmp;

end

function z = shrinkage(Dx, lambda, rho, u)

if norm(Dx - u) <= lambda / rho
    z = 0;
else
    z = (norm(Dx - u) - lambda / rho) * ((Dx - u) ./ norm(Dx - u));
end

end