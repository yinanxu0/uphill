from uphill.document.mixins.internal import InternalMixin
from uphill.document.mixins.plot import PlotMixin
from uphill.document.mixins.serialization import SerializableMixin



class AllMixin(
    SerializableMixin,
    InternalMixin,
    PlotMixin,
):
    ...