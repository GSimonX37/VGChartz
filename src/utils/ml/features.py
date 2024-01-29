import pandas as pd


def countries(data: pd.DataFrame) -> pd.Series:
    """
    Вычисляет количество стран, в которых осуществлялась продажа видеоигры;

    :param data: набор данных;
    :return: количество стран, в которых осуществлялась продажа видеоигры.
    """

    return data.iloc[:, -4:].sum(axis=1)


def same(data: pd.DataFrame) -> pd.Series | bool:
    """
    Вычисляет, является ли разработчик и издатель одной компанией;

    :param data: набор данных;
    :return: является ли разработчик и издатель одной компанией.
    """

    return data['publisher'] == data['developer']


def generate(data: pd.DataFrame) -> pd.DataFrame:
    """
    Генерирует для набора данных следующие признаки:

    - countries: количество стран, в которых осуществлялась продажа видеоигры;
    - same: является ли разработчик и издатель одной компанией;

    :param data: набора данных, для которого будут сгенерированы признаки;
    :return: новый набор данных со сгенерированными признаками.
    """

    features = data.copy()

    features.insert(
        loc=features.shape[1],
        column='countries',
        value=countries(features)
    )

    features.insert(
        loc=features.shape[1],
        column='same',
        value=same(features)
    )

    return features
