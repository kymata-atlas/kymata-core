from typing import Union

from numpy import int_, str_, float_


# Set consistent dtypes for use in arrays and as method outputs
# (to be strict and consistent)
HexelDType = int_
SensorDType = str_
LatencyDType = float_
TransformNameDType = str_

# For method inputs
# (to be permissive)
Hexel = Union[int, HexelDType]
Sensor = Union[str, SensorDType]
Channel = Union[Hexel, Sensor]
Latency = Union[float, LatencyDType]
