# HW1

Installation: 
~~~
pip3 install .
~~~
Аналитика данных:
~~~
data_report
~~~
Использование по этапам
~~~
data_processing --config_path config/type1.yaml
train --config_path config/type1.yaml
eval_model --config_path config/type1.yaml
predict --config_path config/type1.yaml
~~~
Полный пайплайн (без predict). Data preprocessing, train, eval
~~~
train_pipeline --config_path config/type1.yaml
~~~
Запуск тестов
~~~
pytest
~~~

# Структура проекта

------------

    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── artifacts          <- Folder for store temporeal artefacts like splitted data, models and like this
    │
    ├── configs            <- List of configs for train and pytest
    │
    ├── data               <- Data for train model and for tests
    │
    ├── model              <- Scripts for train, eval, predict, data_processing and train_pipeline
    │
    ├── notebooks          <- Jupyter notebooks. EDA and GenerateTestData
    │
    ├── reports            <- Script for get report aboud data
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │
    ├── tests              <- modules tests for pytest
    │   ├── test_data      <- tmp data for tests
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
--------