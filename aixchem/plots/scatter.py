import pandas as pd

from pathlib import Path
import yaml

import matplotlib.pyplot as plt
import matplotlib.markers as mpl_markers
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable


from aixchem.plots.core import Plot


class Scatter(Plot):

    def __init__(self, x, y, colors=None, colormap=None, markers=None, markermap=None, sizes=None, sizeminmax=None, alpha=.8, cbar=True, **kwargs):
        super().__init__(**kwargs)

        self.x = x  # Column that contains the x values
        self.y = y  # Column that contains the y values
        self.c = colors  # Column that contains the values that determine the color of the markers
        self.m = markers  # Column that contain the values that determine the marker style
        self.s = sizes  # Column that contains the values that determine the size of the markers

        self.colormap = colormap  # Dict that maps values in self.c to colors
        self.markermap = markermap  # Dict that maps values in self.m to markers
        self.sizeminmax = sizeminmax  # Tuple that contains the min and max size of the markers
        self.alpha = alpha  # Alpha value of the markers

        sc = self._scatter(self.x, self.y, markers=self.m, c=self.c, cmap=self.colormap, s=self.s, alpha=self.alpha)

        if cbar:
            self.add_cbar(sc)

    def _scatter(self, x, y, markers, edgecolor='none', **kwargs):

        # Create the scatter with uniform marker styles.
        sc = self.ax.scatter(x, y, edgecolor=edgecolor, **kwargs)

        # Apply the marker style afterwards
        if markers is not None:
            # Get marker object for all markers
            markers = [mpl_markers.MarkerStyle(m) for m in markers if not isinstance(m, mpl_markers.MarkerStyle)]
            # transform marker objects to a path
            paths = [m.get_path().transformed(m.get_transform()) for m in markers]
            # Apply path to scatter
            sc.set_paths(paths)

        
        return sc


if __name__ == "__main__":

    from aixchem.test.data import regression_df, classification_df

    X, y = classification_df()


    print(X["feature_0"].min())

    plot = Scatter(
        x=X["feature_0"], 
        y=X["feature_1"], 
        colors=y["target"].map({0: "red", 1: "blue", 2:"green"}),
        #sizes=X["feature_2"],
        title="TEST 1.0")


    plot.show()
