from setuptools import find_packages, setup


with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name="model",
    packages=find_packages(),
    version="0.1.0",
    description="Example of mlops project",
    author="Your name (or your organization/company/team)",
    install_requires=required,
)
