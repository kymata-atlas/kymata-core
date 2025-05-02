from json import JSONEncoder
from typing import Any

from numpy import integer, floating, ndarray

from kymata.entities.expression import ExpressionPoint


class NumpyJSONEncoder(JSONEncoder):
    """
    A JSON encoder for use with Numpy datatypes.
    """
    def default(self, obj):
        if isinstance(obj, integer):
            return int(obj)
        if isinstance(obj, floating):
            return float(obj)
        if isinstance(obj, ndarray):
            return obj.tolist()
        return super().default(obj)


def serialise_expression_point(point: ExpressionPoint) -> dict[str, Any]:
    """
    Serialise an expression point to a dictionary suitable to pass to json.dumps(..., cls=NumpyJSONEncoder).

    Args:
        point (ExpressionPoint): The point to serialise.

    Returns:
        dict: A dictionary suitable to pass to json.dumps(..., cls=NumpyJSONEncoder)
    """
    return {
        "channel": point.channel,
        "latency": point.latency,
        "transform": point.transform,
        "logp_value": point.logp_value,
    }
