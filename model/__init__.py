"""Model definitions for multi-sensor GRF estimation."""

__all__ = ["MultiScaleGRFNet"]
__version__ = "0.1.0"


def __getattr__(name):
    if name == "MultiScaleGRFNet":
        from .grfnet_multiscale import MultiScaleGRFNet

        return MultiScaleGRFNet
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
