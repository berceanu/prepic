"""
The class from which all others inherit

"""
from unyt import allclose_units
from prepic._util import todict, flatten_dict
import logging

logger = logging.getLogger(__name__)


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
        if isinstance(other, type(self)):
            self_vars = flatten_dict(todict(self))
            other_vars = flatten_dict(todict(other))
            common_vars = self_vars.keys() & other_vars.keys()

            for key in common_vars:
                self_val = self_vars[key]
                other_val = other_vars[key]
                if not allclose_units(self_val, other_val, 1e-5):
                    logger.warning(f"Difference in {key}: {self_val} vs {other_val}")
                    return False
            return True
        return False


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)-8s - %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
