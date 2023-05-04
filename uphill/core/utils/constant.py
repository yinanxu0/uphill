import math


##################
# constant value #
##################
CHUNK_SIZE = 4096
INT16MAX = 32768
EPSILON = 1e-7 # to avoid Omission error on some device
LOG_EPSILON = math.log(EPSILON)
DEFAULT_PADDING_VALUE = 0  # used for custom attrs

