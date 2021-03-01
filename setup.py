from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='lapixdl',
    packages=find_packages(exclude=['tests']),
    version='0.7',
    description='Utils for Computer Vision Deep Learning research',

    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lapix-ufsc/lapixdl",

    author='LAPiX',
    license='MIT',

    install_requires=['numpy', 'tqdm', 'seaborn', 'pandas', 'matplotlib'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'cv2'],
    test_suite='tests',

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
