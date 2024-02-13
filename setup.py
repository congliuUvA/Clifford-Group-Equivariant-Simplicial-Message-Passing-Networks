from setuptools import setup

setup(
    name="engineer",
    install_requires=[],
    packages=["engineer"],
    version="0.0.1",
    author="",
    entry_points={
        "console_scripts": [
            "discover_tests = engineer.test.discover_tests:main",
            "sweep = engineer.sweep.sweep:main",
            "sweep_local = engineer.sweep.sweep_local:main",
        ],
    },
)
