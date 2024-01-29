import datetime

from bs4 import BeautifulSoup

from config.parser.managers.parsing import PARSING_FIELDS
from parser.game import Game


class ParsingManager:
    """
    Менеджер парсинга, задачами которого являются:

    - парсинг полученных данных;
    - учет успешно и неуспешно спарсенных данных;

    :var success: успешно спарсенные данные;
    :var failed: неуспешно спарсенные данные.
    """

    def __init__(self):
        self.success: dict[str: int] = {field: 0 for field in PARSING_FIELDS}
        self.failed: dict[str: int] = {field: 0 for field in PARSING_FIELDS}

    async def parse(self, table: str) -> list[Game]:
        """
        Осуществляет парсинг основных данных;

        :param table: данные для парсинга;
        :return: None.
        """

        games = []

        soup = BeautifulSoup(table, 'html.parser')
        rows = (soup
                .find('div', id='generalBody')
                .find('table')
                .find_all('tr')[3:])

        for row in rows:
            game = Game()
            game.name = await self.name(row)
            game.platform = await self.platform(row)
            game.publisher = await self.publisher(row)
            game.developer = await self.developer(row)
            game.vgc = await self.vgc(row)
            game.critic = await self.critic(row)
            game.user = await self.user(row)
            game.shipped = await self.shipped(row)
            game.total = await self.total(row)
            game.america = await self.america(row)
            game.europe = await self.europe(row)
            game.japan = await self.japan(row)
            game.other = await self.other(row)
            game.date = await self.date(row)

            games.append(game)

        return games

    async def name(self, soup: BeautifulSoup) -> str | None:
        """
        Осуществляет парсинг названия видеоигры;

        :param soup: объект BeautifulSoup;
        :return: название видеоигры.
        """

        try:
            name = (soup
                    .find_all('td')[2]
                    .find('a')
                    .text
                    .strip())

            self.success['name'] += 1
            return name
        except AttributeError:
            self.failed['name'] += 1
        except ValueError:
            self.failed['name'] += 1
        except IndexError:
            self.failed['name'] += 1

    async def platform(self, soup: BeautifulSoup) -> str | None:
        """
        Осуществляет парсинг игровой платформы;

        :param soup: объект BeautifulSoup;
        :return: название видеоигры.
        """

        try:
            platform = (soup
                        .find_all('td')[3]
                        .find('img')['alt'])
            self.success['platform'] += 1

            return platform
        except AttributeError:
            self.failed['platform'] += 1
        except ValueError:
            self.failed['platform'] += 1
        except IndexError:
            self.failed['platform'] += 1

    async def publisher(self, soup: BeautifulSoup) -> str | None:
        """
        Осуществляет парсинг издателя;

        :param soup: объект BeautifulSoup;
        :return: издатель.
        """

        try:
            publisher = (soup
                         .find_all('td')[4]
                         .text
                         .strip())

            self.success['publisher'] += 1
            return publisher
        except AttributeError:
            self.failed['publisher'] += 1
        except ValueError:
            self.failed['publisher'] += 1
        except IndexError:
            self.failed['publisher'] += 1

    async def developer(self, soup: BeautifulSoup) -> str | None:
        """
        Осуществляет парсинг разработчика;

        :param soup: объект BeautifulSoup;
        :return: разработчик.
        """

        try:
            developer = (soup
                         .find_all('td')[5]
                         .text
                         .strip())

            self.success['developer'] += 1
            return developer
        except AttributeError:
            self.failed['developer'] += 1
        except ValueError:
            self.failed['developer'] += 1
        except IndexError:
            self.failed['developer'] += 1

    async def vgc(self, soup: BeautifulSoup) -> float | None:
        """
        Осуществляет парсинг оценки VGChartz.com;

        :param soup: объект BeautifulSoup;
        :return: оценка VGChartz.com.
        """

        try:
            vgc = (soup
                   .find_all('td')[6]
                   .text)
            vgc = round(float(vgc), 1)

            self.success['vgc'] += 1
            return vgc
        except AttributeError:
            self.failed['vgc'] += 1
        except ValueError:
            self.failed['vgc'] += 1
        except IndexError:
            self.failed['vgc'] += 1

    async def critic(self, soup: BeautifulSoup) -> float | None:
        """
        Осуществляет парсинг оценки критиков;

        :param soup: объект BeautifulSoup;
        :return: оценка критиков.
        """

        try:
            critic = (soup
                      .find_all('td')[7]
                      .text)
            critic = round(float(critic), 1)

            self.success['critic'] += 1
            return critic
        except AttributeError:
            self.failed['critic'] += 1
        except ValueError:
            self.failed['critic'] += 1
        except IndexError:
            self.failed['critic'] += 1

    async def user(self, soup: BeautifulSoup) -> float | None:
        """
        Осуществляет парсинг оценки пользователей;

        :param soup: объект BeautifulSoup;
        :return: оценка пользователей.
        """

        try:
            user = (soup
                    .find_all('td')[8]
                    .text)
            user = round(float(user), 1)

            self.success['user'] += 1
            return user
        except AttributeError:
            self.failed['user'] += 1
        except ValueError:
            self.failed['user'] += 1
        except IndexError:
            self.failed['user'] += 1

    async def shipped(self, soup: BeautifulSoup) -> float | None:
        """
        Осуществляет парсинг количества отданных копий;

        :param soup: объект BeautifulSoup;
        :return: количество отданных копий.
        """

        try:
            shipped = (soup
                       .find_all('td')[9]
                       .text)
            shipped = float(shipped.replace('m', ''))

            self.success['shipped'] += 1
            return shipped
        except AttributeError:
            self.failed['shipped'] += 1
        except ValueError:
            self.failed['shipped'] += 1
        except IndexError:
            self.failed['shipped'] += 1

    async def total(self, soup: BeautifulSoup) -> float | None:
        """
        Осуществляет парсинг общего количество проданных копий;

        :param soup: объект BeautifulSoup;
        :return: общее количество проданных копий.
        """

        try:
            total = (soup
                     .find_all('td')[10]
                     .text)
            total = float(total.replace('m', ''))

            self.success['total'] += 1
            return total
        except AttributeError:
            self.failed['total'] += 1
        except ValueError:
            self.failed['total'] += 1
        except IndexError:
            self.failed['total'] += 1

    async def america(self, soup: BeautifulSoup) -> float | None:
        """
        Осуществляет парсинг количества проданных копий в Америке;

        :param soup: объект BeautifulSoup;
        :return: количество проданных копий в Америке.
        """

        try:
            america = (soup
                       .find_all('td')[11]
                       .text)
            america = float(america.replace('m', ''))

            self.success['america'] += 1
            return america
        except AttributeError:
            self.failed['america'] += 1
        except ValueError:
            self.failed['america'] += 1
        except IndexError:
            self.failed['america'] += 1

    async def europe(self, soup: BeautifulSoup) -> float | None:
        """
        Осуществляет парсинг количества проданных копий в Европе;

        :param soup: объект BeautifulSoup;
        :return: количество проданных копий в Европе.
        """

        try:
            europe = (soup
                      .find_all('td')[12]
                      .text)
            europe = float(europe.replace('m', ''))

            self.success['europe'] += 1
            return europe
        except AttributeError:
            self.failed['europe'] += 1
        except ValueError:
            self.failed['europe'] += 1
        except IndexError:
            self.failed['europe'] += 1

    async def japan(self, soup: BeautifulSoup) -> float | None:
        """
        Осуществляет парсинг количества проданных копий в Японии;

        :param soup: объект BeautifulSoup;
        :return: количество проданных копий в Японии.
        """

        try:
            japan = (soup
                     .find_all('td')[13]
                     .text)
            japan = float(japan.replace('m', ''))

            self.success['japan'] += 1
            return japan
        except AttributeError:
            self.failed['japan'] += 1
        except ValueError:
            self.failed['japan'] += 1
        except IndexError:
            self.failed['japan'] += 1

    async def other(self, soup: BeautifulSoup) -> float | None:
        """
        Осуществляет парсинг количества проданных копий в остальном мире;

        :param soup: объект BeautifulSoup;
        :return: количество проданных копий в Океании.
        """

        try:
            other = (soup
                     .find_all('td')[14]
                     .text)
            other = float(other.replace('m', ''))

            self.success['other'] += 1
            return other
        except AttributeError:
            self.failed['other'] += 1
        except ValueError:
            self.failed['other'] += 1
        except IndexError:
            self.failed['other'] += 1

    async def date(self, soup: BeautifulSoup) -> str | None:
        """
        Осуществляет парсинг даты выхода;

        :param soup: объект BeautifulSoup;
        :return: дата выхода.
        """

        try:
            date = (soup
                    .find_all('td')[15]
                    .text)

            date = (date
                    .replace('th', '')
                    .replace('st', '')
                    .replace('rd', '')
                    .replace('nd', ''))

            date = date.split()
            if int(date[2]) > datetime.date.today().year % 100:
                date[2] = '19' + date[2]
            else:
                date[2] = '20' + date[2]
            date = ' '.join(date)

            date = (datetime
                    .datetime
                    .strptime(date, '%d %b %Y')
                    .strftime('%Y-%m-%d'))

            self.success['date'] += 1
            return date
        except AttributeError:
            self.failed['date'] += 1
        except ValueError:
            self.failed['date'] += 1
        except IndexError:
            self.failed['date'] += 1

    @staticmethod
    async def page(text: str) -> int:
        """
        Осуществляет парсинг номера последней страницы;

        :param text: данные для парсинга;
        :return: номер последней страницы;
        """

        soup = BeautifulSoup(text, 'html.parser')

        number = (soup
                  .find('div', id='mainContainerSub')
                  .find('div', id='generalBody')
                  .find_all('table')[1]
                  .find_all('th')[0]
                  .text)

        number = (number
                  .split()[1]
                  .replace('(', '')
                  .replace(')', '')
                  .replace(',', ''))

        if int(number) % 50 == 0:
            number = int(number) // 50
        else:
            number = int(number) // 50 + 1

        return number
