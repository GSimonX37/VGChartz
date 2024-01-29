import asyncio

from config.parser.parser import PARAMS
from config.parser.parser import VALID_ATTEMPTS
from parser.game import Game
from parser.managers.file import FileManager
from parser.managers.network.network import NetworkManager
from parser.managers.output import OutputManager
from parser.managers.parsing import ParsingManager
from parser.managers.progress import ProgressManager


class Parser:
    """
    Программа, осуществляющая сбор, обработку и хранение данных;

    :var file: файловый менеджер;
    :var network: сетевой менеджер;
    :var output: менеджер вывода;
    :var parsing: менеджер парсинга;
    :var progress: менеджер прогресса;
    :var stopped: флаг остановки трансфера данных.
    """

    def __init__(self):
        self.file: FileManager = FileManager()
        self.network: NetworkManager = NetworkManager()
        self.output: OutputManager = OutputManager()
        self.parsing: ParsingManager = ParsingManager()
        self.progress: ProgressManager = ProgressManager()
        self.stopped: bool = False

    async def connect(self) -> int:
        """
        Проверяет соединение с сервером перед началом сбора данных;

        :return: Код статуса запроса.
        """

        code = await self.network.connect()

        return code

    async def scrape(self) -> None:
        """
        Запускает:

        - процесс сбора данных;
        - трансфер менеджеру вывода параметров остальных менеджеров;
        - отображение текущего состояния сбора данных;

        :return: None.
        """

        self.progress.starting()

        tasks = [
            asyncio.create_task(self.run()),
            asyncio.create_task(self.transfer(True)),
            asyncio.create_task(self.output.state(True))
        ]

        await asyncio.gather(*tasks)

    async def run(self) -> None:
        """
        Запускает процесс сбора данных;

        :return: None
        """
        current, last = self.progress.progress

        for page in range(current, last + 1, 10):
            if page + 10 <= last:
                pages = range(page, page + 10)
            else:
                pages = range(page, last + 1)

            tables = await self.tables([*pages])

            data = [game.csv() for table in tables for game in table if game]

            await self.file.write(data)

            await self.progress.next(pages.stop - pages.start)

            await self.save()

        self.file.delete()

        await self.transfer()
        await self.output.state()

        self.output.stopped = True
        self.stopped = True

    async def table(self, page: int) -> list[Game]:
        """
        Получает данные, размещенные на странице;

        :param page: номер страницы;
        :return: данные, размещенные на странице.
        """

        link = f'{self.network.url}/games/games.php'

        code, text = None, ''

        while code != 200:
            response = await self.network.get(link, PARAMS | {'page': page})
            code = response['code']

            if code == 200:
                text = response['text']

        games = await self.parsing.parse(text)

        return games

    async def tables(self, pages: list):
        """
        Получает данные, размещенные на страницах;

        :param pages: номера страниц;
        :return: данные, размещенные на странице.
        """

        tasks = []
        for page in pages:
            tasks.append(asyncio.create_task(self.table(page)))

        games = await asyncio.gather(*tasks)

        return games

    async def page(self) -> None | int:
        """
        Получает номер последней страницы;

        :return: номер последней страницы.
        """

        code, number, attempts = None, None, 0
        link = f'{self.network.url}/games/games.php'

        while code != 200 and attempts < VALID_ATTEMPTS:
            response = await self.network.get(link, PARAMS)
            code, attempts = response['code'], attempts + 1

            if code == 200:
                text = response['text']
                number = await self.parsing.page(text)

        return number

    async def disconnect(self) -> None:
        """
        Закрывает сессию;

        :return: None.
        """

        await self.network.session.close()

    async def setting(self,
                      span: tuple[int, int],
                      factor: int,
                      threshold: int,
                      file: str,
                      mode: str,
                      timeout: int,
                      checkpoint: str):
        """
        Настраивает менеджеры;

        :param span: диапазон задержки;
        :param factor: масштаб задержки;
        :param threshold: порог смены типа задержки;
        :param file: имя файла с данными;
        :param mode: режим работы с файлом;
        :param timeout: задержка между выводами текущего состояния;
        :param checkpoint: имя файла контрольной точки в формате json;
        :return: None.
        """

        self.network.setting(span, factor, threshold)
        self.file.setting(file, mode, checkpoint)

        last = await self.page()

        self.progress.setting([1, last])
        self.output.setting(timeout)

        await self.transfer()

    async def save(self) -> None:
        """
        Записывает контрольную точки в формат json;

        :return: None.
        """

        settings = {}
        settings |= self.progress.json()
        settings |= self.network.json()
        settings |= self.output.json()

        await self.file.save(settings)

    async def load(self, checkpoint: str) -> None:
        """
        Читает контрольную точки в формате json;

        :param checkpoint: контрольная точка;
        :return: None.
        """

        settings = self.file.load(checkpoint)

        self.file.setting(
            file=settings['file'],
            mode='a',
            checkpoint=checkpoint
        )

        self.network.setting(
            span=settings['span'],
            factor=settings['factor'],
            threshold=settings['threshold']
        )

        last = await self.page()

        self.progress.setting(
            progress=[settings['progress'], last]
        )

        self.output.setting(
            timeout=settings['timeout']
        )

        await self.transfer()

    async def state(self) -> None:
        """
        Выводит текущее состояние на экран;

        :return: None.
        """

        await self.output.state()

    async def transfer(self, repeat: bool = False) -> None:
        """
        Трансфер менеджеру вывода параметров остальных менеджеров;

        :return: None.
        """

        while not self.stopped:
            await self.output.file(
                file=self.file.file,
                size=self.file.size,
                records=self.file.records,
            )

            await self.output.network(
                statuses=self.network.statuses,
                traffic=self.network.traffic,
                span=await self.network.delay.current(),
            )

            await self.output.parsing(
                success=self.parsing.success,
                failed=self.parsing.failed,
            )

            await self.output.progress(
                passed=self.progress.passed(),
                finish=self.progress.finished,
                speed=self.progress.speed,
                interval=self.progress.interval
            )

            if not repeat:
                break

            await asyncio.sleep(1)
