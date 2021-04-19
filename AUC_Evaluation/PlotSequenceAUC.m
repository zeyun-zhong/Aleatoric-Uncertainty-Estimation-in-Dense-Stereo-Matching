function [] = PlotSequenceAUC(AUC_list, method_names, should_save, file_path, title_name)

if nargin == 4
    title_name = '';
end

% Necessary for correct visualisation
legend_list = strrep(method_names,'_','\_');

% Sort the auc list w.r.t first row, i.e. optimal auc values
AUC_list = sortrows(AUC_list,1);

figure();
for i = 1:size(AUC_list,2)
    plot(AUC_list(:,i),'-','LineWidth',1)
    hold on
end
grid on
xlabel('sequence numbers')
ylabel('AUC value')
xlim([1, size(AUC_list, 1)])
legend(legend_list,'Location','northwest');
title(['Comparison of AUC values', ' ', title_name])
hold off

if (should_save)
    file_path = strcat(file_path, 'AUC_sequence_plot');
    saveas(gcf,file_path,'pdf');
end

end