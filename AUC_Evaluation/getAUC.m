function [auc] = getAUC(percentage_list,error_rate)

auc = 1/2 * percentage_list(1) * error_rate(1); % calculate the first area under the curve, which is actually a rectangle

for i = 2:length(percentage_list) % start from 2
    auc = auc + 1/2 * (percentage_list(i) - percentage_list(i-1)) * (error_rate(i) + error_rate(i-1)); % the rest parts are right trapezoids
end

end

