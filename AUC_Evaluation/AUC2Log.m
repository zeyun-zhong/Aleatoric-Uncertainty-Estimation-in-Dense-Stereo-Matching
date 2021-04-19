function [] = AUC2Log(AUC_values, AUC_average, method_names, should_log, log_file_path)

method_names = cellfun(@(x) replace(x, "-", "_"), method_names, 'UniformOutput', false);

% Combine data
data = cell(size(AUC_values, 1) + 2, size(AUC_values, 2) + 1);
data(1,1) = {'Idx'};
data(1,2:end) = method_names;
data(2,1) = {'average'};
data(2,2:end) = num2cell(AUC_average);
data(3:end,:) = num2cell([linspace(1, size(AUC_values, 1), size(AUC_values, 1))' AUC_values]);

% Convert cell to a table and use first row as variable names
data = cell2table(data(2:end,:), 'VariableNames', data(1,:));
 
% Write the table to a CSV file
if (should_log)
    log_file_path = strcat(log_file_path, 'AUC_values.csv');
    writetable(data, log_file_path);
end

display(data(1,:));

end