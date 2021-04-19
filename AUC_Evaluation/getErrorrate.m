function [error_rate] = getErrorrate(mark_point,conf_list,delta,T)

% initialize the error rate list
error_rate = zeros(length(mark_point),1);

for i = 1:length(mark_point)
    % get the position of the pixel from conf_list
    pixel = conf_list(1:mark_point(i),1:2); 
    % transfer the position of pixel to index
    idx = sub2ind(size(delta),pixel(:,1), pixel(:,2)); 
    % compare the values in delta with threshold
    k = sum(delta(idx) <= T(idx)); 
    % calculate the error rate
    error_rate(i) = 1 - k/mark_point(i); 
end

end

