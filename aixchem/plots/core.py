import pandas as pd

from pathlib import Path
import yaml

import matplotlib.pyplot as plt
import matplotlib.markers as mpl_markers
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable


STYLESHEET = Path(__file__).parent / "style.mpl"


class Plot:

    plt.style.use(STYLESHEET)

    def __init__(self, title=None, figsize=(10, 10), ax=None):

        # Convert freedom units to cm
        figsize = (figsize[0] / 2.54, figsize[1] / 2.54)

        if ax is not None:
            self.fig, self.ax =  ax.get_figure(), ax
        else:
            self.fig, self.ax = plt.subplots(1, 1, figsize=figsize)

        self.set_title(title)

    def show(self):
        plt.show()

    def save(self, path, close=True):

        path = Path(path)

        if path.suffix == ".png":
            self.fig.savefig(path, format="png", dpi=400, bbox_inches='tight')
        elif path.suffix == ".svg":
            self.fig.savefig(path, format="svg", bbox_inches='tight')

        if close:
            plt.close(self.fig)
    
    def set_title(self, title):
        self.ax.set_title(title)

    def set_axis_style(self, xlabel=None, ylabel=None, xfmt=None, yfmt=None):
        
        if xlabel is not None:
            self.ax.set_xlabel(xlabel)
        if ylabel is not None:
            self.ax.set_ylabel(ylabel)

        # fmt e.g.'%.1f'
        if xfmt is not None:
            self.ax.xaxis.set_major_formatter(FormatStrFormatter(xfmt))
        if yfmt is not None:
            self.ax.yaxis.set_major_formatter(FormatStrFormatter(yfmt))

    def add_cbar(self, obj, label=None, size="5%", pad=0.05):
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes("right", size=size, pad=pad)

        # Add color bar
        cbar = self.fig.colorbar(obj, label=label, cax=cax)
        cbar.outline.set_edgecolor('none')
        cbar.ax.minorticks_off()




