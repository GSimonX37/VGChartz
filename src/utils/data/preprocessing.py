import numpy as np
import pandas as pd

from config.paths import FILE_PREPROCESSED_PATH
from config.paths import FILE_RAW_PATH


def preprocessing(file: str) -> None:
    """
    Обрабатывает данные:

    - удаляет явные дубликаты;
    - удаляет записи, которые в поле platform имеют значения "All" или "Series";
    - заменяет значение Unknown в поле "developer" и "publisher" на NaN;

    :param file: полное имя файла с очищенными данными в формате csv;
    :return: None.
    """

    name = fr'{FILE_RAW_PATH}\{file}'
    df = pd.read_csv(name)

    # Удаление явных дубликатов.
    df = df.drop_duplicates()

    # Удаление записей,
    # у которых в поле platform имеются значения "All" или "Series"
    df = df[~df['platform'].isin(['All', 'Series'])]

    # Замена значений Unknown в поле "developer" и "publisher" на NaN.
    df['developer'] = df['developer'].replace({'Unknown': np.nan})
    df['publisher'] = df['publisher'].replace({'Unknown': np.nan})

    df.to_csv(
        path_or_buf=fr'{FILE_PREPROCESSED_PATH}\{file}',
        sep=',',
        index=False
    )
