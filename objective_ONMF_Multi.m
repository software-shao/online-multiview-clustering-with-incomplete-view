function [o] = objective_ONMF_Multi(X,W,U,V,U_star,alpha,beta)
num_views = numel(X);
o = 0;
U_star = U_star(1:size(X{1},1),:);
for i = 1:num_views
    W{i} = W{i}(1:size(X{i},1), 1:size(X{i},1));
    U{i} = U{i}(1:size(X{i},1),:);
    o = o + norm(W{i}*(X{i} - U{i}*V{i}'), 'fro')^2 + alpha(i)*norm(W{i}*(U{i} - U_star), 'fro')^2 + beta(i)*sum(sum(abs(U{i})));
end