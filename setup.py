import setuptools

with open("readme.md") as f:
    long_description = f.read()

setuptools.setup(
    name='ur_control',
    version='0.0.0',
    author='Rasmus Laurvig Haugaard',
    author_email='rasmus.l.haugaard@gmail.com',
    description='Control primitives on top of ur_rtde.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/RasmusHaugaard/ur_control',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'ur_rtde',
        'transform3d',
    ],
    python_requires='>=3.6',
)
