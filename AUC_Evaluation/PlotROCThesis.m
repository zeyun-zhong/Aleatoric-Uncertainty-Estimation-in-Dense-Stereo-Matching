function [] = PlotROCThesis(percentage_opt_good,error_opt_good,percentage_opt_other,error_opt_other,percentage_good,mean_error_rate_good,percentage_other,mean_error_rate_other,lim_y_good,lim_y_other,savefig,legends)
x0=300;
y0=100;
width=700;
height=200;
% [0.4660 0.6740 0.1880]
C = {'b','r','g','m','k','g'}; % Cell array of colros.
markers = {'-*', '-o', '-s','-^'}; % cell array of markers
shape = size(mean_error_rate_good);

figure('DefaultAxesFontSize', 11)
subplot(1,2,1);
plot([0, percentage_opt_good], [0, error_opt_good], '-', 'LineWidth',1, 'color',C{1})
hold on
for i = 1:shape(2)
    plot(percentage_good, [0, mean_error_rate_good(:, i)'], markers{i},'LineWidth',1, 'color',C{i+1})
end
hold off
grid on
xlabel('density [%]')
ylabel('error [%]')
xlim([0, 100])
ylim(lim_y_good)
set(gca,'position',[0.06,0.2,0.35,0.77]);

subplot(1,2,2);
plot([0, percentage_opt_other], [0, error_opt_other], '-', 'LineWidth',1, 'color',C{1})
hold on
for i = 1:shape(2)
    plot(percentage_other, [0, mean_error_rate_other(:, i)'], markers{i},'LineWidth',1, 'color',C{i+1})
end
hold off
grid on
xlabel('density [%]')
%ylabel('error [%]')
xlim([0, 100])
ylim(lim_y_other)
set(gca,'position',[0.46,0.2,0.35,0.77]);

leg = legend(legends,'Location','NorthEastOutside');
legend boxoff
leg.Position = [0.83,0.25,0.15,0.6];

set(gcf,'position',[x0,y0,width,height]);
if savefig
   img_name = 'ROC_thesis.svg';
   saveas(gcf, img_name)
end

end

