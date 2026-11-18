from .MinstedSimulator import MinstedSimulator

__all__ = ["MinstedSimulator", "BayesianMinstedSimulator", "MinstedExposure", "CircularBayesianMinsted", "CircularScan"]


def __getattr__(name):
    # Keep SciPy optional for callers using only the original simulator.
    if name in {"BayesianMinstedSimulator", "MinstedExposure"}:
        from .BayesianMinstedSimulator import BayesianMinstedSimulator, MinstedExposure

        globals().update(
            BayesianMinstedSimulator=BayesianMinstedSimulator,
            MinstedExposure=MinstedExposure,
        )
        return globals()[name]
    if name in {"CircularBayesianMinsted", "CircularScan"}:
        from .CircularBayesianMinsted import CircularBayesianMinsted, CircularScan

        globals().update(CircularBayesianMinsted=CircularBayesianMinsted, CircularScan=CircularScan)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
