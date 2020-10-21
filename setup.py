from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='lapixdl',
    packages=find_packages(include=['lapixdl']),
    version='0.2.0',
    description='Evaluation metrics for segmentation, detection and classification Deep Learning algorithms',

    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lapix-ufsc/lapixdl",

    author='LAPIX',
    license='MIT',

    install_requires=['numpy'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
