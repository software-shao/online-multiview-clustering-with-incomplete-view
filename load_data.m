function [Data, info] = load_data(data_name, random, incomplete_rate)
if strcmp(data_name,'digit')
    D = cell(5,1);
    D{1} = load(strcat('digit/mfeat-fou.txt'));
    D{2} = load(strcat('digit/mfeat-fac.txt'));
    D{3} = load(strcat('digit/mfeat-pix.txt'));
    D{4} = load(strcat('digit/mfeat-mor.txt'));
    D{5} = load(strcat('digit/mfeat-zer.txt'));
    num_views = 5;
    label = zeros(size(D{1},1),1);
    for i=1:10
        label(1+(i-1)*200:200*i) = i;
    end
    if random==1
        p = randperm(numel(label));
        label = label(p);
        for i = 1:num_views
            D{i} = D{i}(p,:);
        end
        Data = D;
        info.size = numel(label);
        info.label= label;
        info.num_views = num_views;
    else
        
    end
    % based on the incomplete rate, make the data incomplete.
    if incomplete_rate > 0
        p = randperm(numel(label)-1);
        p = p+1;
        p = repmat(p, 1,num_views);
        index = cell(num_views,1);
        incomplete_size = ceil(incomplete_rate*numel(label));
        for i = 1:num_views
            index{i} = p((i-1)*incomplete_size+1:i*incomplete_size);
            index{i} = sort(index{i});
        end
        for i = 1:num_views
            for j = 1:numel(index{i})
                total = sum(Data{i}(1:index{i}(j)-1,:),1);
                incomplete = sum(Data{i}(index{i}(1:j-1),:),1);
                Data{i}(index{i}(j),:) = (total - incomplete)./(index{i}(j)-j);
            end
        end
        info.incomplete_index = index;
    end
    info.num_views = 5;
end
end