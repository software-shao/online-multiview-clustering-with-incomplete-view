function [U_total,V, U_star_total, Loss,average_nmi,average_ac] = ONMF_Multi_PGD_search(X, W, option, block_size)
    num_passes = option.pass;
    num_views = numel(X);
    total = size(X{1}, 1);
    label = option.label;
    alpha = option.alpha;
    beta = option.beta;
    skip_loss = option.loss;
    maxIter = option.maxiter;
    
    U_star_total = cell(num_passes,1);
    k = option.k;
    tol = option.tol;
    num_feature = zeros(num_views,1);
    V = cell(num_views,1);
    for i = 1:num_views
        num_feature(i) = size(X{i},2);
        V{i} = rand(num_feature(i), k);
    end
    A = cell(num_views, 1);
    B = cell(num_views, 1);
    num_block = ceil(total/block_size);
    Loss = zeros(num_passes, num_block);
    
    average_nmi = zeros(num_passes, 2);
    average_ac = zeros(num_passes, 2);
    X_sofar = cell(num_views,1);
    
    for pass = 1:num_passes
        t1 = tic;       
        if pass == 1
            U_total = cell(num_views,1);
            U_star_total{pass} = zeros(size(X{i},1),k);
            for i = 1:num_views
                U_total{i} = zeros(size(X{i},1),k);
                A{i} = zeros(size(V{i}));
                B{i} = zeros(size(V{i},2));
            
            end
        else
            U_star_total{pass} = U_star_total{pass-1};
        end
        for i = 1:num_views
            X_sofar{i} = [];
        end
       
        for block_index = 1:num_block
            data_range = (block_index-1)*block_size+1:block_index*block_size;
            U = cell(num_views,1);
            for i =1:num_views
                U{i} = rand(block_size, k);
            end
            U_star = rand(block_size,k);
            if block_index==num_block
                data_range = (block_index-1)*block_size+1:total;
                for i = 1:num_views
                    U{i} = rand(total-(num_block-1)*block_size,k);
                end
                U_star = rand(total-(num_block-1)*block_size,k);
            end
            % Get data blocks, W blocks.
            X_block = cell(num_views,1);
            W_block = cell(num_views,1);
            for i = 1:num_views
                X_block{i} = X{i}(data_range,:);
                W_block{i} = W{i}(data_range, data_range);
                if(total > size(X_sofar{i},1))
                    X_sofar{i} = [X_sofar{i}; X_block{i}];
                end
            end
            % Get initial U and V;
            if pass == 1
                if(block_index==1) 
                    % alternating initialize U and V
                    for j = 1:num_views
                          U{j} = rand(size(U{j}));
                          V{j} = rand(size(V{j}));
                    end
                else
                    % Only initialize U
                    WW = W_block;
                    for j = 1:num_views
                        WW{i} = W_block{i}'*W_block{i};
                        U{j} = rand(size(U{j}));
                    end
                end
                % Initialize U*
                CU = alpha(1)*(W_block{1}.^2)*U{1};
                CC = alpha(1)*(W_block{1}.^2);
                for i = 2:num_views
                    CU = CU + alpha(i)*(W_block{i}.^2)*U{i};
                    CC = CC + alpha(i)*(W_block{i}.^2);
                end
                CC_inv = diag(1./diag(CC));
                U_star = CC_inv*CU; 
            else
                % Only use previous U and U_star
                for i = 1:num_views
                    U{i} = U_total{i}(data_range,:);
                end
                U_star = U_star_total{pass}(data_range,:);
            end
            
            iter = 0;
            converge =0;
            log_out = objective_ONMF_Multi(X_sofar,W,U_total,V,U_star_total{pass},alpha, beta);
            % Fix U_star, update U and V 
            while(iter<maxIter && converge == 0)
                WW = W_block;
                for i = 1:num_views
                    WW{i} = W_block{i}'*W_block{i};
                end
                for i = 1:num_views
                    % calculate U
                        log = norm(W_block{i}*(X_block{i} - U{i}*V{i}'), 'fro')^2 ...
                            +alpha(i)*norm(W_block{i}*(U{i} - U_star), 'fro')^2 + beta(i)*sum(sum(abs(U{i})));
                        diff = 1;
                        gamma_u = 1;
                        search_iter = 0;
                        while diff > 0
                            gradient_u = 2*WW{i}*U{i}*V{i}'*V{i} - 2 * WW{i}*X_block{i}*V{i} + 2*alpha(i)*WW{i}*U{i} - 2*alpha(i)*WW{i}*U_star + beta(i);% + beta(i)*diag(1./D)*U{i};
                            Hessian_u = (2*V{i}'*V{i} + 2 * alpha(i)* eye(size(V{i},2)));
                            U_new{i} = U{i};
                            if sum(sum( WW{i} - eye(size(U{i},1)) ~= 0)) > 0
                                for ii = 1:size(U_new{i},1)
                                    U_new{i}(ii,:) = max(0, U{i}(ii,:) - gamma_u*gradient_u(ii,:)/(WW{i}(ii,ii).*Hessian_u)); % + beta(i)*(inv_norm_u - inv_norm_u^3*uu)));
                                end
                            else
                                U_new{i} = max(0, U{i} - gamma_u*gradient_u/Hessian_u);
                            end
                            log_new = norm(W_block{i}*(X_block{i} - U_new{i}*V{i}'), 'fro')^2 ...
                            +alpha(i)*norm(W_block{i}*(U_new{i} - U_star), 'fro')^2 + beta(i)*sum(sum(abs(U_new{i})));
                            diff = log_new - log;
                            gamma_u = gamma_u/2;
                            if search_iter > 250
                                break;
                            end
                            search_iter = search_iter + 1;
                        end
                      
                        U{i} = U_new{i};
                        % update V
                        n_sofar = size(X_block{i},1)+block_size*(block_index-1);
                        U_total{i}(data_range,:) = U{i};
                        log = norm(W{i}(1:n_sofar,1:n_sofar)*(X_sofar{i}(1:n_sofar,:) - U_total{i}(1:n_sofar,:)*V{i}'), 'fro')^2; 
                        diff = 1;
                        gamma_v = 1;
                        search_iter = 0;
                        while diff > 0
                            gradient_v = -2*(A{i} + X_block{i}'*WW{i}*U_new{i}) + 2*V{i}*(B{i} + U_new{i}'*WW{i}*U_new{i});
                            Hessian_v = 2*(B{i} + U_new{i}'*WW{i}*U_new{i});
                            V_new{i} = max(V{i} - gamma_v*gradient_v/Hessian_v,0);
                            
                            log_new = norm(W{i}(1:n_sofar,1:n_sofar)*(X_sofar{i}(1:n_sofar,:) - U_total{i}(1:n_sofar,:)*V_new{i}'), 'fro')^2; 
                            diff = log_new - log;
                            gamma_v = gamma_v/2;
                            if search_iter > 250
                                break;
                            end
                            search_iter = search_iter + 1;
                        end
                        
                        V{i} = V_new{i};
                        norms = sum(abs(V_new{i}),1);
                        norms = max(norms,1e-20);
                        V_new{i} = V_new{i}./repmat(norms,size(V{i},1),1);
                        U_new{i} = U_new{i}.*repmat(norms,size(U{i},1),1);
                        U{i} = U_new{i};
                        V{i} = V_new{i};                    
                end
                % update U_star
                CU = alpha(1)*(W_block{1}.^2)*U_new{1};
                CC = alpha(1)*(W_block{1}.^2);
                for i = 2:num_views
                    CU = CU + alpha(i)*(W_block{i}.^2)*U_new{i};
                    CC = CC + alpha(i)*(W_block{i}.^2);
                end
                CC_inv = diag(1./diag(CC));
                U_star_new = CC_inv*CU; 
                
                U = U_new;
                V = V_new;
                U_star = U_star_new;
                log_out_new = objective_ONMF_Multi(X_sofar,W,U_total,V,U_star_total{pass},alpha, beta);
                if abs(log_out_new- log_out)/log_out < tol
                    converge = 1;
                end
                log_out = log_out_new;
                iter = iter + 1;
            end
            for i = 1:num_views
                A{i} = A{i} + X_block{i}'*W_block{i}.^2*U{i};
                B{i} = B{i} + U{i}'*W_block{i}.^2*U{i};
            end
            fprintf('Finish the pass %d, block_index %d, within %d iterations, log is %d\n',pass, block_index,iter, log);
            
            U_star_total{pass}(data_range,:) = U_star;
            for i = 1:num_views
                U_total{i}(data_range,:) = U{i};
            end
            if skip_loss == 0
                Loss(pass, block_index) = log_out/size(X_sofar{1},1);  
            end
            if mod(block_index, 50) == 0
                fprintf('Block %d\n', block_index);
            end
        end
        fprintf('Done with pass %d in %d seconds\n', pass, toc(t1));
        replicates = 20;
        tmp_nmi = zeros(replicates,1);
        tmp_ac = zeros(replicates,1);
        for i = 1:replicates
            result_tmp = litekmeans(U_star_total{pass}, k, 'Replicates',20);
            [tmp_ac(i), tmp_nmi(i), ~] = CalcMetrics(label, result_tmp);
        end
        average_nmi(pass, 1) = mean(tmp_nmi);
        average_ac(pass, 1) = mean(tmp_ac);
        fprintf('Done with pass %d\n', pass);
        fprintf('The nmi is %d and the AC is %d\n', average_nmi(pass,1), average_ac(pass, 1));
    end