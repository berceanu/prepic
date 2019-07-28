"""
The class from which all others inherit

"""
from unyt import allclose_units
from prepic._util import todict, flatten_dict
import logging

logger = logging.getLogger(__name__)


class BaseClass:
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
