from setuptools import find_packages, setup


with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name="model",
    packages=find_packages(),
    version="0.1.0",
    description="Example of mlops project",
    author="Your name (or your organization/company/team)",
    entry_points={
        "console_scripts": [
            "train_pipeline = model.train_pipeline:main",
            "train = model.train:main",
            "eval_model = model.eval:main",
            "predict = model.predictor:main",
            "data_processing = model.data_preprocessing:main",
            "data_report = reports.data_report:main",
        ]
    },
    install_requires=required,
)