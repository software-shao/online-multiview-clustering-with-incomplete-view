clear
warning off;
rng(1);

addpath('mulNMF/');
addpath('mulNMF_incomplete_2/');

data_set_name = 'digit';
block_size = 50;
random = 1;
num_clusters = 10;
maxiter = 200;
decay = 1;
tol = 1e-4;
num_passes = 2;
skip_loss = 0;
incomplete_range = 0.4:0.2:0.4;
average_nmi = cell(numel(incomplete_range),1);
average_ac = cell(numel(incomplete_range),1);
nmf_ac = zeros(numel(incomplete_range),20);
nmf_nmi = nmf_ac;

[Data, info] = load_data(data_set_name, random, 0.4);
num_views = info.num_views;
alpha = 1e-2 * ones(num_views,1);
beta = 1e-7 * ones(num_views,1);
for i = 1:num_views
    Data{i}(Data{i}<0) = 0;
    Data{i} = Data{i}./sum(sum(Data{i}));
end

% Construct W
W = cell(numel(Data),1);
if exist('info.W')
    for i = 1:numel(Data)
        W{i} = info.W{i};
    end
    info.W = [];
else
    for i = 1:numel(Data)
        W{i} = diag(sparse(ones(info.size,1)));
        if exist('info.incomplete_index')
            counter = 0;
            for j = 1:numel(info.incomplete_index{i})
                counter = counter +1;
                W{i}(info.incomplete_index{i}(j),info.incomplete_index{i}(j)) = 1.0*counter/info.incomplete_index{i}(j);
            end
        end
    end
end
option.label = info.label;
option.k = num_clusters;
option.maxiter = maxiter;
option.tol = tol;
option.num_cluster = num_clusters;
option.decay = decay;
option.alpha = alpha;
option.beta = beta;
option.pass = num_passes;
option.loss = skip_loss;
t1 = tic;
[U_total,V, U_star, Loss, average_nmi, average_ac] = ONMF_Multi_PGD_search(Data, W, option, block_size);
display(average_nmi(:,1)');
display(average_ac(:,1)');
toc(t1)



