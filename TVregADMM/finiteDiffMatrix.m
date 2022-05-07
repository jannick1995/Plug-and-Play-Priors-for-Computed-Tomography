function D = finiteDiffMatrix(n)
%
% finiteDiffMatrix computes the global first order finite difference matrix
% D.
%
% Input: 
%
%          n - number of pixels in one direction (assumed to be square)
%
% Output:
%
%          D - global first order finite difference matrix
%
% Author: Jannick Jørgensen Lønver

% Take the size of a square image (n by n)

% Construct bi-diagonal matrix
e = ones(n,1);
A = spdiags([-e e],0:1,n-1,n);
FD = [A;zeros(1,n)];

D1 = kron(FD,speye(n)); % horizontal
D2 = kron(speye(n),FD); % vertical

D3 = [D1;D2];

% Permute the rows such that they coincide with horizontal, vertical for
% each pixel
idx = zeros(2*n*n,1);
idx(1:2:end) = 1:n*n;
idx(2:2:end) = (n*n+1):2*n*n;

D = D3(idx,:);

end