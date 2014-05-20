# -*- coding: utf-8 -*-
from enaml.layout.layout_helpers import grid
#from enaml.widgets.api import GroupBox


def auto_grid_layout(self):
    """
    """
    children = self.widgets()
    labels = children[::2]
    widgets = children[1::2]
    n_labels = len(labels)
    n_widgets = len(widgets)
    if n_labels != n_widgets:
        if n_labels > n_widgets:
            labels.pop()
        else:
            widgets.pop()

    return [grid(labels, widgets)]
