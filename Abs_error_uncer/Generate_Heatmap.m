function [] = Generate_Heatmap(points, savefig, savename, unc_limits, label_x, label_y)
    grid_number = 128;   %refinement of map
    minvals = min(points);
    maxvals = max(points);
    rangevals = maxvals - minvals;
    xidx = 1 + round((points(:,1) - minvals(1)) ./ rangevals(1) * (grid_number-1));
    yidx = 1 + round((points(:,2) - minvals(2)) ./ rangevals(2) * (grid_number-1));
    density = accumarray([yidx, xidx], 1, [grid_number,grid_number]);  %note y is rows, x is cols
    density = (1.0 / size(points,1)) * density * 100;

    f = figure('Name', 'Histogram ', 'DefaultAxesFontSize', 11);
    imagesc(density, 'xdata', [minvals(1), maxvals(1)], 'ydata', [minvals(2), maxvals(2)])
    grid on
    xlim([0 200])
    ylim(unc_limits)
    if label_x
        xlabel('Absolute Disparity Error [pixel]')
    end
    
    if label_y
        %ylabel('Standard Deviation [pixel]')
        ylabel('SD [pixel]')
    end
    %ylabel('Confidence')
    set(gca,'YDir','normal')
    colormap(flipud(hot))
    caxis(gca,[10^-3 1])
    set(gca,'ColorScale','log')
    cb=colorbar('Location', 'northoutside');
    set(cb, 'Ticks', [0, 10^-3, 10^-2, 10^-1, 10^0], 'TickLabels', {'0%', '0.001%', '0.01%', '0.1%', '1%'})
    %set(cb, 'Ticks', [0, 10^-2, 10^-1, 10^0, 10^1], 'TickLabels', {'0%', '0.01%', '0.1%', '1%', '10%'})
    cb.Label.Interpreter = 'latex';
    width=280;
    height=200;
    set(gca,'position',[0.155,0.18,0.805,0.6]);
    x0=300;
    y0=100;
    
    set(gcf,'position',[x0,y0,width,height]);
    % txt = ['R^2: ', num2str(Rsq, '%.4f')];
    % text(160,280,txt);
    %txt_ = ['\rho: ', num2str(roh(2, 1), '%.4f')];
    %text(165,270,txt_);
    if savefig
        saveas(gcf,savename)
    end
end

                                                                                                                                                                                                                                                