function [sub_list_best,sub_list_worst] = sub_error_rate(sub_list,conf_list,delta,T)

for i = 1:length(sub_list)
    sub_percentage_list = sub_list{i};
    mark_point = ceil(sub_percentage_list * length(conf_list));
    
%     if mark_point(1) == 0
%         mark_point(1) = 1;
%     end
    
    % get the position of the pixel from conf_list
    pixel = conf_list(1:mark_point(end),1:2); 
    % transfer the position of pixel to index
    idx = sub2ind(size(delta),pixel(:,1), pixel(:,2)); 
    % compare the values in delta with threshold
    k = delta(idx) <= T(idx);
    
    sub_k = k(mark_point(1)+1:end);
    
    % error rate for best sub list
    best_sub_list = sort(sub_k,'descend');
    k(mark_point(1)+1:end) = best_sub_list;
    best_correctness_list = k;
    
    % error rate for worst sub list
    worst_sub_list = sort(sub_k);
    k(mark_point(1)+1:end) = worst_sub_list;
    worst_correctness_list = k;
    
    best_error_rate = [];
    worst_error_rate = [];
    
    for j = 2:length(mark_point)-1
        
        B = best_correctness_list(1:mark_point(j));
        best_error_rate(j-1) =1 - sum(B)/length(B);
        
        W = worst_correctness_list(1:mark_point(j));
        worst_error_rate(j-1) =1 - sum(W)/length(W);
        
    end
    
    sub_list_best{i,1} = [sub_percentage_list(2:end-1);best_error_rate];
    sub_list_worst{i,1} = [sub_percentage_list(2:end-1);worst_error_rate];

end

