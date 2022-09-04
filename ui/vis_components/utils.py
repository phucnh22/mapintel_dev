import colorcet as cc  # use glasbey_light colormap for dark background and categorical values
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, rgb2hex
from sklearn.preprocessing import LabelEncoder


def val_to_color(col, cmap=cc.cm.glasbey_light):
    """Converts a column of values to hex-type colors"""
    norm = Normalize(vmin=col.min(), vmax=col.max(), clip=True)
    mapper = ScalarMappable(norm=norm, cmap=cmap)
    rgba = mapper.to_rgba(col)

    return np.apply_along_axis(rgb2hex, 1, rgba)


def cat_to_color(col, cmap=cc.cm.glasbey_light):
    """Converts a column of categorical values to hex-type colors"""
    new_col = LabelEncoder().fit_transform(col)
    return val_to_color(new_col, cmap)
