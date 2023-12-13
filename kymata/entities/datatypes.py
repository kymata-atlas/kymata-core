from numpy import int_, str_, float_


Hexel = int  # Todo: change this and others to `type Hexel = int` on dropping support for python <3.12
Sensor = str
Latency = float

# Set consistent dtypes for use in arrays
HexelDType = int_
SensorDType = str_
LatencyDType = float_
FunctionNameDType = str_
