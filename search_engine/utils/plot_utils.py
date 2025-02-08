from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

def plot_whiskers(df, x_col, y_cols, labels, colors, linestyles, log_scale=False, show_legend=True, ncol=1,
                  bbox_to_anchor=None, figsize=(10, 8), resultname= None, xlabel="", ylabel="", 
                  yticks1 = None, yticks2=None,lowerb = 0.25, upperb=0.75, rects = [], lines = []):
    fig, ax = plt.subplots(figsize=figsize)
    for i, col in enumerate(y_cols):
        med = df.groupby(x_col)[col].median().values
        lower = df.groupby(x_col)[col].quantile(lowerb).values
        upper = df.groupby(x_col)[col].quantile(upperb).values
        x = np.arange(len(med))
        yerr = [med - lower, upper - med]
        ax.errorbar(x, med, yerr=yerr, capsize=5, color=colors[i], fmt='', linestyle=linestyles[i], label=labels[i])
        ax.plot(x, med, linestyle='None', marker='.', markersize=10, color=colors[i])
    if show_legend:
        ax.legend(bbox_to_anchor=bbox_to_anchor, ncol=ncol)
    if log_scale:
        ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, df.groupby(x_col).count().index)
    if yticks1 is not None:
        ax.set_yticks(yticks1, yticks2)
    
    
    # Add rectangle
    for rect in rects:
        ax.add_patch(rect)
    
    for line in lines:
        ax.add_line(line)
#     rect = Rectangle((-0.3, 16.5), 3.5, 1, color='white', clip_on=False,zorder=10)
    

#     # Add slash lines
#     line1 = Line2D([-0.3, 3+0.3], [17.5, 17.5], color='black', linewidth=1, ls="--", clip_on=False,zorder=11)
#     line2 = Line2D([-0.3, 3+0.3], [16.5, 16.5], color='black', linewidth=1, ls="--", clip_on=False,zorder=11)
#     line1 = Line2D([0-0.1, 0+0.1], [17, 18], color='black', linewidth=1, clip_on=False,zorder=11)
#     line2 = Line2D([0-0.1, 0+0.1], [16, 17], color='black', linewidth=1, clip_on=False,zorder=11)
    
#     line3 = Line2D([2-0.1, 2+0.1], [17, 18], color='black', linewidth=1, clip_on=False,zorder=11)
#     line4 = Line2D([2-0.1, 2+0.1], [16, 17], color='black', linewidth=1, clip_on=False,zorder=11)
    
#     line5 = Line2D([1-0.1, 1+0.1], [17, 18], color='black', linewidth=1, clip_on=False,zorder=11)
#     line6 = Line2D([1-0.1, 1+0.1], [16, 17], color='black', linewidth=1, clip_on=False,zorder=11)
    
#     line7 = Line2D([3-0.1, 3+0.1], [17, 18], color='black', linewidth=1, clip_on=False,zorder=11)
#     line8 = Line2D([3-0.1, 3+0.1], [16, 17], color='black', linewidth=1, clip_on=False,zorder=11)
    
#     ax.add_line(line1)
#     ax.add_line(line2)
#     ax.add_line(line3)
#     ax.add_line(line4)
#     ax.add_line(line5)
#     ax.add_line(line6)
#     ax.add_line(line7)
#     ax.add_line(line8)

    if resultname is not None:
        plt.savefig(resultname, bbox_inches = 'tight')
    plt.show()