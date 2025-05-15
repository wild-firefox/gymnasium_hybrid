from setuptools import setup

setup(
    name="gymnasium_hybrid",
    version="0.0.1",  # original gym_hybrid version='0.0.1'
    packages=["gymnasium_hybrid"],
    include_package_data=True,
    package_data={
        "gymnasium_hybrid": ["*.png", "*.jpg"],
    },
    install_requires=[
        "gymnasium>=1.0.0",
        "numpy",
        "pygame>=2.3.0",
        "opencv-python>=4.11.0.86",
        "moviepy>=2.1.2",
    ],
)
