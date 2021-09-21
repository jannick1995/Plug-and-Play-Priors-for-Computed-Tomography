function [x, z, u, info] = TVregADMM(A, b, lambda, rho)
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
%        A      - System matrix
%        b      - Right hand side 
%        lambda - Regularization parameter
%        rho    - penalty parameter
%
% Output:
%
%        x      - Solution from TVregADMM
%
%        info   - Struct containing the following attributes:
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
% Author: Jannick Jørgensen Lønver

t_start = tic;

% Global constants
DISPLAY = 0;
MAXITER = 300;
ABSTOL = 1e-2;
RELTOL = 1e-2;
%mu = 10;
%tau_incr = 2;
%tau_decr = 2;

% Extract dimensions
[~,n2] = size(A); % n2 = n^2

% Initialize z and u
z = zeros(2*n2,1);
u = zeros(2*n2,1);

if ~DISPLAY
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n','iter','|| r ||', ...
        'eps pri','|| s ||', 'eps dual','objective');
end

% Construct the global finite difference matrix D
D = finiteDiffMatrix(sqrt(n2));

for k = 1:MAXITER
    
    % x-update using CGLS (do not form the matrices)
    [x,~] = lsqr([A;sqrt(rho)*D],[b;sqrt(rho)*(z+u)],[],200);
    %x = (AtA + rho * DtD) \ (Atb + rho * D'*(z+u));
    
    % Store Dx
    Dx = D*x;
    
    % Store previous z before updating it
    z_old = z;
    
    % z-update using shrinkage
    for i = 1:2:2*n2
        z(i:i+1) = shrinkage(Dx(i:i+1),lambda,rho,u(i:i+1));
    end
%     parfor i = 1:2*n2
%         if norm(Dx(i)-u(i),2) <= lambda / rho
%             z(i) = 0;
%         else
%             z(i) = (norm(Dx(i)-u(i)) - lambda / rho) * ((Dx(i)-u(i)) / norm(Dx(i)-u(i)));
%         end
%         
%     end
    
    % dual-variable update
    u = u + z - Dx;
    
    % Diagnostics
    info.objval(k) = objective(A, b, lambda, x, z);
    
    info.r_norm(k) = norm(z - Dx);
    info.s_norm(k) = norm(-rho * D'*(z - z_old));
    
    info.eps_pri(k)  = sqrt(2*n2) * ABSTOL + RELTOL * max(norm(D*x),norm(z));
    info.eps_dual(k) = sqrt(n2) * ABSTOL + RELTOL * norm(rho*D'*u);
    
    if ~DISPLAY
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n',k, ...
            info.r_norm(k), info.eps_pri(k), info.s_norm(k), ...
            info.eps_dual(k), info.objval(k))
    end
    
    if (info.r_norm(k) < info.eps_pri(k) && ...
            info.s_norm(k) < info.eps_dual(k))
        break;
    end
    
    %adaptively change rho
%     if info.r_norm(k) > mu * info.s_norm(k)
%         rho = tau_incr * rho;
%     elseif info.s_norm(k) > mu * info.r_norm(k)
%         rho = rho / tau_decr;
%     end
   
end

if ~DISPLAY
    toc(t_start);
end

end 

function p = objective(A, b, lambda, x, z)

p = (1/2) * sum(A*x - b).^2 + lambda * norm(z);

end

function z = shrinkage(Dx, lambda, rho, u)

if norm(Dx - u) <= lambda / rho
    z = 0;
else
    z = (norm(Dx-u) - lambda / rho) * ((Dx-u) ./ norm(Dx-u));
end

end