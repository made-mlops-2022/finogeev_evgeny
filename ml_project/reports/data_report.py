import pandas as pd


def main():
    df = pd.read_csv("data/heart_cleveland_upload.csv")
    print("Размер данных", df.shape[0])
    print("==="*20)

    print("Статистики непрерыных признаков")
    print(df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']].describe())
    print("==="*20)

    print("Соотношения категориальных признаков")
    for col in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']:
        print(df[col].value_counts())
        print("***"*10)
    print("==="*20)

    print("Распределение целевого значения")
    print(df['condition'].value_counts())
    print("==="*20)

    print("Коррелции признаков")
    print(df.corr().round(2))
    print("==="*20)

    print("Пример данных")
    print(df.sample(10))
    print("==="*20)


if __name__ == "__main__":
    main()
