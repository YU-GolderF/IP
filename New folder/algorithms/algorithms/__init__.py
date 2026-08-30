"""Fingerprint enhancement algorithms used by the shared Streamlit app."""

ALGORITHM_STATUS = (
    {"key": "rhlt", "name": "RHLT", "owner": "Member 1", "available": True},
    {"key": "algorithm_2", "name": "Gabor Filter Bank", "owner": "Member 2", "available": True},
    {"key": "algorithm_3", "name": "Histogram Equalization", "owner": "Member 3", "available": True},
    {"key": "algorithm_4", "name": "Unsharp Masking", "owner": "Member 4", "available": True},
)


def get_algorithm_runner(algorithm_name: str):
    """Return the run function for the named algorithm."""
    if algorithm_name == "RHLT":
        from algorithms.rhlt.pipeline import run_rhlt
        return run_rhlt
    elif algorithm_name == "Gabor Filter Bank":
        from algorithms.algorithm_2.pipeline import run_algorithm
        return run_algorithm
    elif algorithm_name == "Histogram Equalization":
        from algorithms.algorithm_3.pipeline import run_algorithm
        return run_algorithm
    elif algorithm_name == "Unsharp Masking":
        from algorithms.algorithm_4.pipeline import run_algorithm
        return run_algorithm
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")


__all__ = ["ALGORITHM_STATUS", "get_algorithm_runner"]

