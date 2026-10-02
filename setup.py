"""Discover the two existing package roots without walking assets or outputs."""

from setuptools import find_packages, setup

setup(
    packages=(
        find_packages(where="src", include=["densekp_retrain", "densekp_retrain.*"])
        + ["densekp"]
        + [f"densekp.{name}" for name in find_packages(where="densekp")]
    )
)
