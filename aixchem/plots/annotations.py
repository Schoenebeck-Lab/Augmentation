import pandas as pd

from pathlib import Path
import yaml

import matplotlib.pyplot as plt
import matplotlib.markers as mpl_markers


class Annotation:

    def __init__(self, ax=None, text=None, xy=(0, 0), color="#004080", size=10, vis=True, fancy=True):
        # Use current ax in case ax was not specified
        self.ax = ax if ax is not None else plt.gca()

        self.artist = ax.annotate(
                self._format(text),
                xy=xy,
                xycoords='data',
                xytext=(0, 3),
                textcoords='offset points',
                color=color,
                size=size,
                va="bottom", ha="center",
                )

        # Set visibility
        self.artist.set_visible(vis)

    @staticmethod
    def _format(text):
        """ Format the text"""
        # Process input
        if type(text) == list:
            # Every list element will be displayed in an individual line
            text = "".join(f"{el}\n" if idx != len(text) - 1 else f"{el}" for idx, el in enumerate(text))
        elif type(text) == dict:
            txt = str()
            for key, val in text.items():
                if key != list(text)[-1]:
                    txt += r"$\bf{" + str(key) + r"}$: " + str(val) + "\n"
                else:
                    txt += r"$\bf{" + str(key) + r"}$: " + str(val)
            text = txt
        else:
            text = r"$\bf{" + str(text) + r"}$"

        return text
