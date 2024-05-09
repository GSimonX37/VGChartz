import os

import pandas as pd

from config.paths import FILE_PREPROCESSED_PATH
from config.paths import MODELS_PATH
from ml.training import train
from utils.explorer import explorer


def main():
    """
    Тока входа тренировки моделей на предварительно обработанных данных;

    :return: None.
    """

    names = explorer(FILE_PREPROCESSED_PATH, '*.csv')
    os.system('cls')
    print('Список предобработанных файлов:', names,
          sep='\n',
          flush=True)

    if name := input('Выберите файл: '):
        data = pd.read_csv(f'{FILE_PREPROCESSED_PATH}/{name}')

        models = {}

        print(flush=True)
        names = explorer(path=MODELS_PATH,
                         ext='*.py',
                         exclude=('__init__.py', 'model.py'))
        print('Список файлов c моделями:', names, sep='\n', flush=True)

        if files := input('Выберите один или несколько файлов: '):
            for file in files.split():
                name = file.split('.')[0]

                modul = __import__(
                    name=f'ml.models.{name}',
                    globals=globals(),
                    locals=locals(),
                    fromlist=['model'],
                    level=0
                )

                models[name] = modul.model

        train(
            models=models,
            data=data,
            n_trials=300,
            n_jobs=4
        )


if __name__ == '__main__':
    main()
