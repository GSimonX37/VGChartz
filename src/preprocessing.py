import os
import pandas as pd

from config.paths import FILE_RAW_PATH
from config.paths import FILE_PREPROCESSED_PATH
from utils.data import prepare
from utils.explorer import explorer


def main():
    """
    Тока входа предварительной обработки данных;

    :return: None.
    """

    names = explorer(FILE_RAW_PATH, '*.csv')
    os.system('cls')
    print('Список необработанных файлов:', names, sep='\n', flush=True)

    if name := input('Выберите файл: '):
        name = name.split('.')[0]
        data = pd.read_csv(f'{FILE_RAW_PATH}/{name}.csv')

        # Подготовка к предварительно обработке данных.
        data = prepare(data)

        # Сохранение предобработанных данных.
        data.to_csv(
            path_or_buf=fr'{FILE_PREPROCESSED_PATH}\{name}.csv',
            sep=',',
            index=False
        )


if __name__ == '__main__':
    main()
