import numpy as np
import pandas as pd


def prepare(data) -> pd.DataFrame:
    """
    Подготавливает данные для предварительно обработки;

    :param data: набор данных.
    :return:
    """

    data = data.copy()

    # Удаление явных дубликатов.
    data = data.drop_duplicates()

    # Удаление записей, со значениями "All" или "Series" в поле platform.
    data = data[~data['platform'].isin(['All', 'Series'])]

    # Замена значений Unknown в поле "developer" и "publisher" на NaN.
    data['developer'] = data['developer'].replace({'Unknown': np.nan})
    data['publisher'] = data['publisher'].replace({'Unknown': np.nan})

    # Отбор данных, которые содержат записи в необходимых полях.
    columns = ['date', 'platform', 'publisher', 'developer', 'total']
    data = data[data[columns].notna().all(axis=1)]

    # Удаление выбросов.
    data = data[data['total'] < 1.0]

    # Добавление данных о продажах в других регионах.
    countries = ['america', 'europe', 'japan', 'other']
    data = data[columns + countries]

    # Удалим записи с нулевой целевой переменной.
    data = data[data['total'] != 0.0]

    # Замена значения NaN на 0 в полях america, europe, japan и other.
    data.loc[:, countries] = data.loc[:, countries].fillna(0)

    # Замена значений, отличные от 0 на True, равные 0, на False.
    data.loc[:, countries] = data.loc[:, countries] != 0.0

    # Сортировка данных по полю "date" и удаление данного поля.
    data = (data
            .sort_values(by='date')
            .reset_index(drop=True)
            .drop(columns=['date']))

    # Сортировка полей.
    data = data[['platform', 'publisher', 'developer',
                 'america', 'europe', 'japan', 'other',
                 'total']]

    return data
