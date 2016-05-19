function [y] = vl_nnbnorm_eval(x,g,b,mu,sigma2)
% VL_NNBNORM  CNN batch normalisation
%  This is the version of the batch normalization function that is used
%  during test time. It assumes that a mean mu and a var sigma2 has been
%  estimated on batches from the training data offline, and their (fixed)
%  values are used during evaluation, skipping the mini-batch mean/var
%  computation
% 
%   Y = VL_NNBNORM(X,G,B,M,S) computes the batch normalization of the
%   input X. This is defined as:
%
%      Y(i,j,k,t) = G(k) * (X(i,j,k,t) - mu(k)) / sigma(k) + B(k)
%
%   where
%
%      mu(k) = mean_ijt X(i,j,k,t),
%      sigma(k) = sqrt(sigma2(k) + EPSILON),
%      sigma2(k) = mean_ijt (X(i,j,k,t) - mu(k))^2

epsilon = 1e-4 ;
sigma   = sqrt(sigma2 + epsilon) ;

x_size = [size(x,1), size(x,2), size(x,3), size(x,4)];
g = reshape(g, [1 x_size(3) 1]) ;
b = reshape(b, [1 x_size(3) 1]) ;
x = reshape(x, [x_size(1)*x_size(2) x_size(3) x_size(4)]) ;

y = bsxfun(@minus, x, mu); % y <- x_mu
y = bsxfun(@plus, bsxfun(@times, g ./ sigma	, y), b) ;
y = reshape(y, x_size) ;