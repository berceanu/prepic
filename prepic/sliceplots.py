# -*- coding: utf-8 -*-

"""Module containing useful 2D plotting abstractions on top of matplotlib."""

import matplotlib.transforms as transforms
import numpy as np
from matplotlib.artist import setp, getp
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from typing import Any, Optional

def _idx_from_val(arr1d, val):
    """Given a 1D array, find index of closest element to given value.

    Parameters
    ----------
    arr1d : arr1d: :py:class:`numpy.ndarray`
        1D array of values.
    val : float
        Element value.

    Returns
    -------
    idx : int
        Element position in the array.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.linspace(10, 20, 11)
    >>> _idx_from_val(a, 14.2)
    4
    """
    idx = (np.abs(arr1d - val)).argmin()
    return idx

class Plot2D:
    r"""Pseudo-color plot of a 2D array with optional 1D slices attached.

    Parameters
    ----------
    fig : :class:`~matplotlib.figure.Figure`, optional
        Empty figure to draw on.
        If ``None``, a new :class:`~matplotlib.figure.Figure` will be created.
        Defaults to ``None``.
    arr2d: :py:class:`np.ndarray`
        Data to be plotted.
    h_axis: :py:class:`np.ndarray`
        Values on the "x" axis.
    v_axis: :py:class:`np.ndarray`
        Values on the "y" axis.
    xlabel: str, optional
        x-axis label.
    ylabel: str, optional
        y-axis label.
    zlabel: str, optional
        Label for :py:class:`~matplotlib.colorbar.Colorbar`.
    kwargs : dict, optional
        Other plot options, see examples below.

    Examples
    --------
    .. plot::
       :include-source:

        import numpy as np
        from matplotlib import pyplot

        from sliceplots import Plot2D

        uu = np.linspace(0, np.pi, 128)
        data = np.cos(uu - 0.5) * np.cos(uu.reshape(-1, 1) - 1.0)

        fig = pyplot.figure(figsize=(8,8))

        Plot2D(
            fig=fig,
            arr2d=data,
            h_axis=uu,
            v_axis=uu,
            xlabel=r"$x$ ($\mu$m)",
            ylabel=r"$y$ ($\mu$m)",
            zlabel=r"$\rho$ (cm${}^{-3}$)",
            hslice_val=0.75,
            vslice_val=2.75,
            hslice_opts={"color": "#1f77b4", "lw": 1.5, "ls": "-"},
            vslice_opts={"color": "#d62728", "ls": "-"},
            cmap="viridis",
            cbar=True,
            extent=(0, np.pi, 0, np.pi),
            vmin=-1.0,
            vmax=1.0,
            text="your text here",
        )
    """

    def __init__(
        self,
        *,
        fig: Optional[Figure] = None,
        arr2d: np.ndarray,
        h_axis: np.ndarray,
        v_axis: np.ndarray,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        zlabel: Optional[str] = None,
        **kwargs: Optional[Any],
    ):
        self.extent = kwargs.get(
            "extent", (np.min(h_axis), np.max(h_axis), np.min(v_axis), np.max(v_axis))
        )
        #
        xmin, xmax, ymin, ymax = self.extent
        xmin_idx, xmax_idx = _idx_from_val(h_axis, xmin), _idx_from_val(h_axis, xmax)
        ymin_idx, ymax_idx = _idx_from_val(v_axis, ymin), _idx_from_val(v_axis, ymax)
        #
        self.data = arr2d[ymin_idx:ymax_idx, xmin_idx:xmax_idx]
        self.min_data, self.max_data = np.amin(self.data), np.amax(self.data)
        self.vmin, self.vmax = (
            kwargs.get("vmin", self.min_data),
            kwargs.get("vmax", self.max_data),
        )
        #
        self.h_axis = h_axis[xmin_idx:xmax_idx]
        self.v_axis = v_axis[ymin_idx:ymax_idx]
        #
        self.label = {"x": xlabel, "y": ylabel, "z": zlabel}
        #
        self.cbar = kwargs.get("cbar", True)
        # see https://matplotlib.org/users/colormapnorms.html
        self.norm = kwargs.get("norm")
        #
        self.hslice_val = kwargs.get("hslice_val")
        self.vslice_val = kwargs.get("vslice_val")
        self.hslice_idx = None
        self.vslice_idx = None
        if self.hslice_val is not None:
            self.hslice_idx = _idx_from_val(self.v_axis, self.hslice_val)
        if self.vslice_val is not None:
            self.vslice_idx = _idx_from_val(self.h_axis, self.vslice_val)
        #
        self.text = kwargs.get("text", "")
        #
        if fig is None:  # make new figure
            self.fig = Figure()
            self.canvas = FigureCanvas(self.fig)
        else:
            self.fig = fig
            self.canvas = self.fig.canvas

        self.im = None  # image to be created by .imshow()

        self.ax0 = None  # main axes
        self.axh = None  # horizontal slice axes
        self.axv = None  # vertical slice axes

        self._draw_fig(**kwargs)

    def _main_panel(self, **kwargs):
        self.im = self.ax0.imshow(
            self.data,
            origin="lower",
            extent=self.extent,
            aspect="auto",
            norm=self.norm,
            interpolation="none",
            cmap=kwargs.get("cmap", "viridis"),
            vmin=self.vmin,
            vmax=self.vmax,
        )
        #
        self.ax0.set_xlabel(self.label["x"])
        self.ax0.set_ylabel(self.label["y"])

    def _draw_fig(self, **kwargs):
        slice_opts = {"ls": "-", "color": "#ff7f0e", "lw": 1.5}  # defaults
        hslice_opts = slice_opts.copy()
        vslice_opts = slice_opts.copy()
        #
        hslice_opts.update(kwargs.get("hslice_opts", {}))
        vslice_opts.update(kwargs.get("vslice_opts", {}))

        # #
        if (self.hslice_idx is None) and (self.vslice_idx is None):
            gs = GridSpec(1, 1, height_ratios=[1], width_ratios=[1])
            self.ax0 = self.fig.add_subplot(gs[0])
            self._main_panel(**kwargs)

        # ---- #
        elif (self.hslice_idx is not None) and (self.vslice_idx is None):
            gs = GridSpec(2, 1, height_ratios=[1, 3], width_ratios=[1])
            self.ax0 = self.fig.add_subplot(gs[1, 0])
            self.axh = self.fig.add_subplot(gs[0, 0], sharex=self.ax0)
            #
            self._main_panel(**kwargs)
            #
            self.ax0.axhline(y=self.v_axis[self.hslice_idx], **hslice_opts)
            #
            trans = transforms.blended_transform_factory(
                self.ax0.get_yticklabels()[0].get_transform(), self.ax0.transData
            )
            self.ax0.text(
                0,
                self.v_axis[self.hslice_idx],
                "{:.1f}".format(self.v_axis[self.hslice_idx]),
                color=hslice_opts["color"],
                transform=trans,
                ha="right",
                va="center",
            )
            #
            self.axh.set_xmargin(0)
            self.axh.set_ylabel(self.label["z"])
            self.axh.plot(self.h_axis, self.data[self.hslice_idx, :], **hslice_opts)
            self.axh.set_ylim(self.vmin, self.vmax)
            #
            self.axh.xaxis.set_visible(False)
            #
            for sp in ("top", "bottom", "right"):
                self.axh.spines[sp].set_visible(False)
            #
            self.fig.subplots_adjust(hspace=0.03)

        # | #
        elif (self.vslice_idx is not None) and (self.hslice_idx is None):
            gs = GridSpec(1, 2, height_ratios=[1], width_ratios=[3, 1])
            self.ax0 = self.fig.add_subplot(gs[0, 0])
            self.axv = self.fig.add_subplot(gs[0, 1], sharey=self.ax0)
            #
            self._main_panel(**kwargs)
            #
            self.ax0.axvline(x=self.h_axis[self.vslice_idx], **vslice_opts)
            #
            trans = transforms.blended_transform_factory(
                self.ax0.transData, self.ax0.get_xticklabels()[0].get_transform()
            )
            self.ax0.text(
                self.h_axis[self.vslice_idx],
                0,
                "{:.1f}".format(self.h_axis[self.vslice_idx]),
                color=vslice_opts["color"],
                transform=trans,
                ha="center",
                va="top",
            )
            #
            self.axv.set_ymargin(0)
            self.axv.set_xlabel(self.label["z"])
            self.axv.plot(self.data[:, self.vslice_idx], self.v_axis, **vslice_opts)
            self.axv.set_xlim(self.vmin, self.vmax)
            #
            self.axv.yaxis.set_visible(False)
            #
            for sp in ("top", "left", "right"):
                self.axv.spines[sp].set_visible(False)
            #
            self.fig.subplots_adjust(wspace=0.03)

        # --|-- #
        else:
            gs = GridSpec(2, 2, height_ratios=[1, 3], width_ratios=[3, 1])
            self.ax0 = self.fig.add_subplot(gs[1, 0])
            self.axh = self.fig.add_subplot(gs[0, 0], sharex=self.ax0)
            self.axv = self.fig.add_subplot(gs[1, 1], sharey=self.ax0)
            #
            self._main_panel(**kwargs)
            #
            self.ax0.axhline(y=self.v_axis[self.hslice_idx], **hslice_opts)  # ##----##
            self.ax0.axvline(x=self.h_axis[self.vslice_idx], **vslice_opts)  # ## | ##
            # --- #
            trans = transforms.blended_transform_factory(
                self.ax0.get_yticklabels()[0].get_transform(), self.ax0.transData
            )
            self.ax0.text(
                0,
                self.v_axis[self.hslice_idx],
                "{:.1f}".format(self.v_axis[self.hslice_idx]),
                color=hslice_opts["color"],
                transform=trans,
                ha="right",
                va="center",
            )
            # | #
            trans = transforms.blended_transform_factory(
                self.ax0.transData, self.ax0.get_xticklabels()[0].get_transform()
            )
            self.ax0.text(
                self.h_axis[self.vslice_idx],
                0,
                "{:.1f}".format(self.h_axis[self.vslice_idx]),
                color=vslice_opts["color"],
                transform=trans,
                ha="center",
                va="top",
            )
            # --- #
            self.axh.set_xmargin(0)  # otherwise ax0 may have white margins
            self.axh.set_ylabel(self.label["z"])
            self.axh.plot(self.h_axis, self.data[self.hslice_idx, :], **hslice_opts)
            self.axh.set_ylim(self.vmin, self.vmax)
            # | #
            self.axv.set_ymargin(0)
            self.axv.set_xlabel(self.label["z"])
            self.axv.plot(self.data[:, self.vslice_idx], self.v_axis, **vslice_opts)
            self.axv.set_xlim(self.vmin, self.vmax)
            # hide the relevant axis
            self.axh.xaxis.set_visible(False)  # -
            self.axv.yaxis.set_visible(False)  # |
            # "Despine" the slice profiles
            for ax, spines in (
                (self.axh, ("top", "bottom", "right")),
                (self.axv, ("top", "left", "right")),
            ):
                #
                for sp in spines:
                    ax.spines[sp].set_visible(False)
            #
            self.fig.subplots_adjust(wspace=0.03, hspace=0.03)
        #
        self.ax0.text(
            0.02, 0.02, self.text, transform=self.ax0.transAxes, color="#ff7f0e"
        )
        #
        if self.cbar:
            cax = inset_axes(self.ax0, width="70%", height="3%", loc=9)
            cbar = self.fig.colorbar(self.im, cax=cax, orientation="horizontal")
            cbar.set_label(self.label["z"], color="#ff7f0e")
            cbar.ax.xaxis.set_ticks_position("top")
            cbar.ax.xaxis.set_label_position("top")
            cbar.ax.tick_params(color="#ff7f0e", width=1.5, labelsize=8)
            cbxtick_obj = getp(cbar.ax.axes, "xticklabels")
            setp(cbxtick_obj, color="#ff7f0e")
