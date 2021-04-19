function [mark_point,percentage_list,sub_list] = percentageList(conf_list,percentage_interval,step)

mark_point = ceil(percentage_interval(2:end) * length(conf_list));
for i = 1:length(mark_point)-1
    n = mark_point(i);
    if conf_list(n,3) ~= 0
        while conf_list(n,3) == conf_list(n+1,3) 
            n = n+1;
        end
        mark_point(i) = n;
    else
        mark_point(i) = mark_point(end);
    end
end
mark_point = unique(mark_point);
percentage_list = mark_point/length(conf_list);

percentage_list = [0, percentage_list];
step_list = diff(percentage_list);

% sub_list contains all sections of the percentage list that are larger
% than a certain threshold (here 10%) and should therefore be subdivided
n = 0;
sub_list = {};
for i = 1:length(step_list)
    if step_list(i) >= 0.1   % this threshold (0.1) represents where the average auc should be calculated
        n = n+1;
        new_step = step_list(i)/round(step_list(i)/step);
        sub_percentage_list = percentage_list(i) : new_step: percentage_list(i+1);
        sub_list{n,1} = sub_percentage_list;
    end
end

percentage_list = percentage_list(2:end);

end

