function [percentage_list, error_rate] = error_list_combine(sub_list_best,sub_list_worst,percentage_list,error_rate)

combined_error_list = [];
for i = 1:length(sub_list_best)
    combined_error_list = [combined_error_list,[sub_list_best{i};sub_list_worst{i}(2,:)]];
end

combined_error_list = [combined_error_list, [percentage_list;error_rate';error_rate']];
combined_error_list = combined_error_list';

combined_error_list = sortrows(combined_error_list);

percentage_list = combined_error_list(:,1);
error_rate = mean(combined_error_list(:,2:3),2);

end

