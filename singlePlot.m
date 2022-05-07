function singlePlot(x1,y1,cm,lw,fs,xlab,ylab,y2,cm2,y3,cm3)

% x1  : values to be plotted on the x-axis
% y1  : values to be plotted on the y-axis
% cm  : colour and marker for item 1
% lw  : linewidth
% fs  : fontsize
% xlab: xlabel
% ylab: ylabel
% y2  : values to be plotted on the y-axis
% cm2 : colour and marker for item 2

if nargin < 8 % If you only need to plot one item
    
    if isempty(cm)
        plot(x1,y1,'linewidth',lw)
    else
        plot(x1,y1,cm,'linewidth',lw)
    end
    
elseif nargin < 10 % Plot two items
    plot(x1,y1,cm,'linewidth',lw)
    hold on
    plot(x1,y2,cm2,'linewidth',lw)
    hold off
else
    plot(x1,y1,cm,'linewidth',lw)
    hold on
    plot(x1,y2,cm2,'linewidth',lw)
    plot(x1,y3,cm3,'linewidth',lw)
end

xlabel(xlab,'fontsize',fs)
ylabel(ylab,'fontsize',fs)
hold off

end
