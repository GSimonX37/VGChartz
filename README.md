![VGChartz](resources/header.jpg)

## Цель и задачи проекта

**Цель проекта**: провести анализ данных, размещенных на сайте 
[VGChartz.com](https://www.vgchartz.com), 
на их основе обучить модель, предсказывающую количество проданных копий 
видеоигры.

**Задачи проекта**:
1. Собрать и систематизировать данные, 
размещенные на сайте [VGChartz.com](https://www.vgchartz.com).
2. Предварительно обработать и провести разведочный анализ данных.
3. Обучить модель, выработать систему по оценки её эффективности.
4. Создать веб-сервис, способный обрабатывать запросы пользователя.

## Этапы проекта 

<table>
    <thead>
        <tr>
            <th>№</th>
            <th>Название этапа</th>
            <th>Описание этапа</th>
            <th>Инструменты</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>1</td>
            <td>Сбор и систематизация данных</td>
            <td>
                Написание программы, осуществляющей сбор и систематизацию данных 
                с сайта <a href="https://www.vgchartz.com">VGChartz.com</a>.
                Сбор и систематизация данных.
            </td>
            <td> 
                <ul>
                    <li>AIOHTTP</li>
                    <li>BeautifulSoup4</li>
                </ul> 
            </td> 
        </tr>
         <tr>
            <td>2</td>
            <td>Разведочный анализ данных</td>
            <td>
                Анализ основных свойств данных, выявление распределений, 
                общих зависимостей и аномалий 
                с помощью инструментов визуализации.
            </td>
            <td> 
                <ul>
                    <li>Jupyter</li>
                    <li>Matplotlib</li>
                    <li>NumPy</li>
                    <li>Pandas</li>
                    <li>Seaborn</li>
                </ul> 
            </td>
        </tr>
        <tr>
        <td>3</td>
            <td>Обучение модели</td>
            <td>
                На основе предварительно обработанных данных, обучение модели, 
                предсказывающей количество проданных копий видеоигры. 
                Выработка системы по оценке эффективности модели.
            </td>
            <td> 
                <ul>
                    <li>Jupyter</li>
                    <li>NLTK</li>
                    <li>NumPy</li>
                    <li>Pandas</li>
                    <li>Seaborn</li>
                    <li>Scikit-learn</li>
                </ul> 
            </td>
        </tr>
        <tr>
            <td>4</td>
            <td>Создание веб-сервиса</td>
            <td>
                Интеграция модели в веб-сервис.
            </td>
            <td> 
                <ul>
                    <li>FastAPI</li>
                    <li>Pandas</li>
                    <li>Uvicorn</li>
                </ul> 
            </td>
        </tr>
    </tbody>
</table>

## Блокноты

1. [exploring.ipynb](notebooks/exploring.ipynb) - предварительная обработка 
и проведение разведочного анализа данных.
2. [training.ipynb](notebooks/training.ipynb) - обучение модели, 
выработка системы по оценки её эффективности.

## Набор данных

Набор данных размещен на сайте 
[kaggle.com](https://www.kaggle.com/datasets/gsimonx37/vgchartz), 
последнюю версию набора данных вы можете найти там.

## Документация

1. [Начало работы](docs/starting.md).
2. [Структура проекта](docs/structure.md).
3. [Описание данных](docs/data.md).
4. [Получение данных](docs/parsing.md).
5. [Предварительная обработка данных](docs/preprocessing.md).
6. [Тренировка и оценка моделей](docs/training.md).
7. [Запуск приложения](docs/application.md).

## Результаты

Наилучшей предсказательной способностью обладает модель **LGBMRegressor**. 
На тестовых данных удалось достичь метрики **RMSE** менее **0.117** (мл. копий).
Присутствует **гетероскедастичность** остатков.

![error](resources/training/error.png)

Кривые обучения показывают, что признак переобучения модели **отсутствует**.
**Добавление новых данных** вероятнее всего **не улучшит**
предсказательную способность модели.
При добавлении новых данных время обучения и время предсказания 
возрастают **линейно**.

![scalability](resources/training/scalability.png)

## Лицензия

Распространяется по лицензии GNU General Public License v3.0. 
См. [LICENSE](LICENSE.txt) для получения дополнительной информации.
