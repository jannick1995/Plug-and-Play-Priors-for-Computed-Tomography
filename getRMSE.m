function out = getRMSE(exact,approx)

% getRMSE computes the Root Mean Squared Error (RMSE) of an approximated
% solution compared to the exact solution.
%
% input:
%
%       exact  - The ground truth (vector)
%
%       approx - The approximated solution (vector)
%
% output:
%
%          out - RMSE of the approximated solution compared to the ground
%          truth
%

exact = exact(:);
approx = approx(:);

m = length(exact);

out = sqrt( (1/m) * sum((exact - approx).^2) );
   
end