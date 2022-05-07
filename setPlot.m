function setPlot(hGraphic)

% Set the interpreter to latex fonts
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');
set(0,'defaulttextInterpreter','latex')

% Sets the axis to specific settings if needed
if nargin == 1
    hAxis = get (hGraphic, 'CurrentAxes');
    set (hAxis,'FontSize',16);
    for hLine = get(hAxis, 'Children')
        set (hLine, 'LineWidth', 1.2);
        set (hLine, 'MarkerSize',8);
    end
end
    
end