from typing import Union

from numpy import int_, str_, float_


# Set consistent dtypes for use in arrays (to be strict and consistent)
HexelDType = int_
SensorDType = str_
LatencyDType = float_
TransformNameDType = str_

# For function signatures (to be permissive)
Hexel = int
Sensor = str
Channel = Union[Hexel, Sensor]
Latency = float
