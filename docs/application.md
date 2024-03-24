# Запуск приложения

Вы можете запустить приложение непосредственно из текущего окружения 
или в контейнере Docker.

## Запуск приложения
Точка входа запуска приложения находится в файле 
[application.py](../src/application.py):

```python
import os

import uvicorn

from app.application import model
from config.paths import TRAINED_MODELS_PATH
from utils.explorer import explorer


def main() -> None:
    """
    Тока входа запуска приложения;

    :return: None.
    """

    names = explorer(TRAINED_MODELS_PATH)
    os.system('cls')
    print('Список моделей:', names, sep='\n', flush=True)

    if directory := input('Выберите модель: '):
        file = TRAINED_MODELS_PATH + rf'\{directory}\{directory}.joblib'
        labels = TRAINED_MODELS_PATH + rf'\{directory}\labels.json'

        model.load(file, labels)

        uvicorn.run(
            app="app.application:app",
            host="127.0.0.1",
            port=8000,
            log_level="info"
        )


if __name__ == "__main__":
    main()
```

Чтобы приложение начало работу, необходимо запустить данный файл. 
Программа отобразит содержимое директории [models](../models), 
где хранятся модели, обученные на этапе тренировки моделей 
(см. [Тренировка моделей](training.md)).

![file](../resources/application/models.jpg)

После выбора модели, приложение начнет свою работу.

![file](../resources/application/run.jpg)

Запрос к приложению осуществляется посредством **post-запроса**, 
в теле которого должно быть указано:
1. date: год выхода видеоигры.
2. platform: игровая платформа.
3. publisher: издатель.
4. developer: разработчик.
5. america: продажи в Америке (True / False).
6. europe: продажи в Европе (True / False).
7. japan: продажи в Японии (True / False).
8. other: продажи в остальном Мире (True / False).

Пример программы, осуществляющей запрос к приложению и получающей ответ от него:

```python
import requests

data = {
    'date': 2020,
    'platform': 'DS',
    'publisher': 'Ubisoft',
    'developer': 'Global A Entertainment',
    'america': True,
    'europe': False,
    'japan': False,
    'other': True
}

r = requests.post('http://127.0.0.1:8000/', json=data)

print(r.content.decode('utf-8'))
```

Тело ответа на запрос содержит:
1. date: год выхода видеоигры.
2. platform: игровая платформа.
3. publisher: издатель.
4. developer: разработчик.
5. america: продажи в Америке (True / False).
6. europe: продажи в Европе (True / False).
7. japan: продажи в Японии (True / False).
8. other: продажи в остальном Мире (True / False).
9. sales: количество проданных копий видеоигры.

## Запуск приложения в контейнере Docker

Чтобы запустить приложение в контейнере Docker необходимо создать образ из файла 
[Dockerfile](../Dockerfile):

```
FROM python:3.10

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

COPY ./models/lgbmr ./models/lgbmr
COPY ./src/app ./app
COPY ./src/run.py .

CMD ["python", "run.py", "lgbmr"]
```

Чтобы создать образ, необходимо выполнить следующую команду:

```commandline
docker build -t application .
```

После создания образа необходимо запустить приложение в контейнере:

```commandline
docker run -d --name heart-disease-serving -p 8000:8000 application
```

[К описанию проекта](../README.md)