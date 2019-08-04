"""
Abstract base classes and interface for prepic.
"""

import warnings

from matplotlib import pyplot
from unyt import allclose_units

from prepic.util import todict, flatten_dict


class BaseClass:
    """Implements equality testing between class instances which inherit it.

    Does a floating point comparison (including unit checking) between common
    instance attributes of two instances of a class that inherits from `BaseClass`.
    The instance attributes are collected recursively, ie. if the child class contains
    sub-classes as attributes, their attributes are collected as well.

    Examples
    --------
    >>> import unyt as u
    >>> class MyClass(BaseClass):
    ...     def __init__(self, attr1, attr2):
    ...         self.attr1 = attr1
    ...         self.attr2 = attr2
    ...
    >>> inst1 = MyClass(attr1=5.2 * u.m, attr2= 3.2 * u.s)
    >>> inst2 = MyClass(attr1=2.5 * u.m, attr2= 3.2 * u.s)
    >>> inst1 == inst2
    False
    >>> inst1 == 5.2 * u.m
    False
    """

    def __eq__(self, other):
        """Overrides the default implementation"""
        if not isinstance(other, type(self)):
            return False

        self_vars = flatten_dict(todict(self))
        other_vars = flatten_dict(todict(other))
        common_vars = self_vars.keys() & other_vars.keys()

        # instances are not equal if any of the common attributes have different values
        for key in common_vars:
            self_val = self_vars[key]
            other_val = other_vars[key]

            if not allclose_units(self_val, other_val, 1e-5):
                warnings.warn(
                    f"Difference in {key}: {self_val} vs {other_val}", RuntimeWarning
                )
                return False

        return True


# Inspired by yellowbrick.
class Plot:
    """
    The root of the visual object hierarchy that defines how sliceplots
    creates, stores, and renders visual artifacts using matplotlib.

    Parameters
    ----------
    ax : matplotlib Axes, default: None
        The axis to plot the figure on. If None is passed in the current axes
        will be used (or generated if required).

    fig : matplotlib Figure, default: None
        The figure to plot the Visualizer on. If None is passed in the current
        plot will be used (or generated if required).

    kwargs : dict
        Keyword arguments that are passed to the base class and may influence
        the visualization as defined in other Visualizers. Optional keyword
        arguments include:

        =============   =======================================================
        Property        Description
        -------------   -------------------------------------------------------
        size            specify a size for the figure
        color           specify a color, colormap, or palette for the figure
        title           specify the title of the figure
        =============   =======================================================

    Notes
    -----
    Plots maintain a reference to an ``ax`` object, a Matplotlib Axes where the
    figures are drawn and rendered, as well as to a ``fig`` object, a Matplotlib
    Figure on which the object will be plotted.
    """

    def __init__(self, ax=None, fig=None, **kwargs):
        self.ax = ax
        self.fig = fig
        self.size = kwargs.pop("size", None)
        self.color = kwargs.pop("color", None)
        self.title = kwargs.pop("title", None)

    @property
    def ax(self):
        """
        The matplotlib axes that the object draws upon (can also be a grid
        of multiple axes objects). The visualizer uses :meth:`~matplotlib.pyplot.gca`
        to create an axes for the user if one has not been specified.
        """
        if not hasattr(self, "_ax") or self._ax is None:
            self._ax = pyplot.gca()
        return self._ax

    @ax.setter
    def ax(self, ax):
        self._ax = ax

    @property
    def fig(self):
        """
        The matplotlib fig that the object draws upon. The visualizer uses
        the matplotlib method :meth:`~matplotlib.pyplot.gcf` to create a figure for
        the user if one has not been specified.
        """
        if not hasattr(self, "_fig") or self._fig is None:
            self._fig = pyplot.gcf()
        return self._fig

    @fig.setter
    def fig(self, fig):
        self._fig = fig

    @property
    def size(self):
        """
        Returns the actual size in pixels as set by matplotlib, or
        the user provided size if available.
        """
        if not hasattr(self, "_size") or self._size is None:
            self._size = self.fig.get_size_inches() * self.fig.dpi
        return self._size

    @size.setter
    def size(self, size):
        self._size = size
        if self._size is not None:
            width, height = size
            width_in_inches = width / self.fig.get_dpi()
            height_in_inches = height / self.fig.get_dpi()
            self.fig.set_size_inches(width_in_inches, height_in_inches)

    def process(self, X, y=None, **kwargs):
        """
        Processes the data to be plotted.

        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features

        y : ndarray or Series of length n
            An array or series of target or class values

        kwargs: dict
            Keyword arguments passed to the drawing functionality.

        Returns
        -------
        self : Plot
            The process method must always return self to support pipelines.
        """
        return self

    def draw(self, **kwargs):
        """
        This function is implemented for developers to hook into the
        matplotlib interface and to create an internal representation of the
        data in the form of a figure or axes.

        Parameters
        ----------

        kwargs: dict
            generic keyword arguments.

        """
        raise NotImplementedError("Subclasses must implement a drawing interface.")

    def finalize(self, **kwargs):
        """
        Finalize executes any subclass-specific axes finalization steps.

        Parameters
        ----------
        kwargs: dict
            generic keyword arguments.

        Notes
        -----
        The user calls poof and poof calls finalize. Developers should
        implement specific finalization methods like setting titles
        or axes labels, etc.
        """
        return self.ax

    def poof(self, outpath=None, clear_figure=False, **kwargs):
        """
        Poof makes the magic happen and a visualizer appear! You can pass in
        a path to save the figure to disk with various backends, or you can
        call it with no arguments to show the figure either in a notebook or
        in a GUI window that pops up on screen.

        Parameters
        ----------
        outpath: string, default: None
            path or None. Save figure to disk or if None show in window

        clear_figure: boolean, default: False
            When True, this flag clears the figure after saving to file or
            showing on screen. This is useful when making consecutive plots.

        kwargs: dict
            generic keyword arguments.

        Notes
        -----
        Developers don't usually override poof, as it is
        primarily called by the user to render the visualization.
        """
        # Ensure that draw has been called
        if self._ax is None:
            warn_message = (
                "{} does not have a reference to a matplotlib.Axes "
                "the figure may not render as expected!"
            )
            warnings.warn(warn_message.format(self.__class__.__name__), UserWarning)

        # Finalize the figure
        self.finalize()

        if outpath is not None:
            pyplot.savefig(outpath, **kwargs)
        else:
            pyplot.show()

        if clear_figure:
            self.fig.clear()

    def set_title(self, title=None):
        """
        Sets the title on the current axes.

        Parameters
        ----------
        title: string, default: None
            Add title to figure or if None leave untitled.
        """
        title = self.title or title
        if title is not None:
            self.ax.set_title(title)
