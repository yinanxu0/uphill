
from .serialization import SerializableMixin
from .algorithm import AlgorithmMixin
from .auto_compress import AutoCompressMixin



class AllMixins(
    SerializableMixin,
    AlgorithmMixin,
    AutoCompressMixin,
):
    ...