import datetime


class Game(object):
    """
    Видеоигра;

    :var name: название;
    :var date: дата выхода;
    :var platform: игровая платформа;
    :var publisher: издатель;
    :var developer: разработчик;
    :var shipped: количество отданных копий;
    :var total: общее количество проданных копий;
    :var america: количество проданных копий в Америке;
    :var europe: количество проданных копий в Европе;
    :var japan: количество проданных копий в Японии;
    :var other: остальные продажи в мире;
    :var vgc: оценка VGChartz.com;
    :var critic: оценка критиков;
    :var user: оценка пользователей;

    """

    def __init__(self):
        self.name: str | None = None
        self.date: datetime.date | None = None
        self.platform: str | None = None
        self.publisher: str | None = None
        self.developer: str | None = None
        self.shipped: int | None = None
        self.total: int | None = None
        self.america: int | None = None
        self.europe: int | None = None
        self.japan: int | None = None
        self.other: int | None = None
        self.vgc: float | None = None
        self.critic: float | None = None
        self.user: float | None = None

    def __bool__(self):
        for attribute in self.__dict__:
            if getattr(self, attribute):
                return True

        return False

    def csv(self) -> list:
        """
        Возвращает данные для записи в csv-файл.

        :return: Данные для записи в csv-файл.
        """

        return [
            self.name,
            self.date,
            self.platform,
            self.publisher,
            self.developer,
            self.shipped,
            self.total,
            self.america,
            self.europe,
            self.japan,
            self.other,
            self.vgc,
            self.critic,
            self.user,
        ]
