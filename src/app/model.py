import joblib
import pandas as pd


class Model(object):
    """
    Модель, предсказывающую количество проданных копий видеоигры;

    :var model: модель.
    """

    def __init__(self):
        self.model = None

    def load(self, model: str) -> None:
        """
        Загружает модель;

        :param model: полное имя файла модели в формате .joblib;
        :return: None.
        """

        self.model = joblib.load(model)

    def result(self, data: pd.DataFrame) -> float:
        """
        Предсказывает количество проданных копий видеоигры по следующим данным:

        - date: год выхода видеоигры;
        - platform: игровая платформа;
        - publisher: издатель;
        - developer: разработчик;
        - america: продажи в Америке (True / False);
        - europe: продажи в Европе (True / False);
        - japan: продажи в Японии (True / False);
        - other: продажи в остальном Мире (True / False);

        :param data: данные видеоигры;
        :return: количество проданных копий видеоигры.
        """

        predict = self.model.predict(data)
        predict = round(predict[0], 2)

        return predict
