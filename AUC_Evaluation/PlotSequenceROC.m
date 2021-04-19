function [percentage_opt,error_opt,percentage,mean_error_rate] = PlotSequenceROC(error_rate, method_names, lim_y, savefig, legends, region_name)

if nargin == 5
    region_name = '';
end

shape = size(error_rate);
len = length(error_rate{1});
mean_error_rate = zeros(len, length(method_names)-1);
step = 1/len;
percentage = 0:step:1;
percentage = percentage * 100;
method_names = cellfun(@(x) strrep(strrep(strrep(strrep(x, 'CVA-Net_', ''), '_2', ''), '-paper', ''), '_', '-'),method_names,'UniformOutput',false);

x0=300;
y0=100;
width=350;
height=200;
% [0.4660 0.6740 0.1880]
C = {'b','r','g','k','m','g'}; % Cell array of colros.
markers = {'-*', '-o', '-^', '-s'}; % cell array of markers

for i = 1:shape(1)
    figure('visible','off', 'DefaultAxesFontSize', 11)
    error_seq = cell2mat(error_rate(i, 1));
    epsilon = error_seq(end);
    percentage_opt = 1-epsilon:0.001:1;
    error_opt = (percentage_opt - (1 - epsilon)) ./ percentage_opt;
    plot([0, percentage_opt], [0, error_opt], '-', 'LineWidth',1.5, 'color',C{1})
    hold on
    for j = 1:shape(2)
        error = cell2mat(error_rate(i, j));
        mean_error_rate(:, j) = mean_error_rate(:, j) + error';
       
        plot(percentage, [0, error], markers{j},'LineWidth',1.5,'color',C{j+1})
    end
    hold off
    grid on
    xlabel('density')
    ylabel('error')
    xlim([0, 1])
    %ylim(lim_y)
    legend(legends,'Location','northwest');
    title(['ROC Curve of Sequence ', int2str(i), ' ', region_name])
    set(gca,'position',[0.12,0.16,0.85,0.75]);
    set(gcf,'position',[x0,y0,width,height]);
    if savefig
       %saveas(gcf, ['ROC_', int2str(i),'_',region_name, '.svg']) 
    end
end

% plot mean roc curve of all sequences
mean_error_rate = mean_error_rate / shape(1);
epsilon = mean_error_rate(end, 1);
percentage_opt = 1-epsilon:0.001:1;
error_opt = (percentage_opt - (1 - epsilon)) ./ percentage_opt;
error_opt = error_opt * 100;
percentage_opt = percentage_opt * 100;
mean_error_rate = mean_error_rate * 100;

figure('DefaultAxesFontSize', 11)
plot([0, percentage_opt], [0, error_opt], '-', 'LineWidth',1, 'color',C{1})
hold on
for i = 1:shape(2)
    plot(percentage, [0, mean_error_rate(:, i)'], markers{i},'LineWidth',1, 'color',C{i+1})
end
hold off
grid on
xlabel('density [%]')
ylabel('error [%]')
xlim([0, 100])
ylim(lim_y)
legend(legends,'Location','northwest');
%title(['Mean ROC Curve', ' ', region_name])

set(gca,'position',[0.1,0.2,0.85,0.78]);
set(gcf,'position',[x0,y0,width,height]);
if savefig
   img_name = ['ROC_mean','_',region_name, '.svg'];
   saveas(gcf, img_name)
end