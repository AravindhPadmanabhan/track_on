from setuptools import setup, find_packages

setup(
    name="trackon",           # Name of your package
    version="0.1.0",             # Version
    packages=find_packages(),    # Automatically find all packages
    install_requires=[],
    author="Your Name",
    author_email="you@example.com",
    description="A short description of what your package does",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/my_project",  # optional
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
