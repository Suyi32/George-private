from setuptools import find_packages, setup
setup(
    name="testbedlib",
    packages=find_packages(include=["testbedlib", "testbedlib.simulator", "testbedlib.util"]),
    version="0.1.0",
    description="Testbed library",
    author="Suyi",
    license="MIT",
)