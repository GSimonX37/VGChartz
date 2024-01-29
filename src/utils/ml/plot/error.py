import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.gridspec import GridSpec
from sklearn.metrics import root_mean_squared_error


def error(y_true: pd.Series,
          y_predict: pd.Series,
          title: str,
          path: str = None) -> None:
    """
    Строит график ошибок регрессионной модели;

    :param y_true: фактические значения целевой переменной;
    :param y_predict: прогнозируемые значения целевой переменной;
    :param title: заголовок графика;
    :param path: имя директории, в которую необходимо сохранить график;
    :return: None.
    """

    # Создаем объект фигуры.
    figure = plt.figure(
        layout='constrained',
        figsize=(20, 5)
    )

    # Определяем заголовок для фигуры.
    figure.suptitle(
        t=title,
        fontsize='x-large',
        y=1.05
    )

    # Создаем сетку для размещения графиков.
    gird = GridSpec(
        figure=figure,
        nrows=1,
        ncols=2,
        wspace=0.025
    )

    # Добавляем оси для графиков в сетку.
    figure.add_subplot(gird[0, 0])
    figure.add_subplot(gird[0, 1])

    sns.scatterplot(
        x=y_predict.values,
        y=y_true.values,
        sizes=8,
        ax=figure.axes[0],
        color=sns.color_palette('hls', 15)[8]
    )

    # Добавляем линию идеальной модели.
    figure.axes[0].axline(
        xy1=(0, 0),
        xy2=(1, 1),
        color='grey',
        linestyle=':',
        label='Идеальная модель'
    )

    # Определяем стиль для графика 1.
    # Определяем заголовок.
    figure.axes[0].set_title(
        label='Фактические и прогнозируемые значения',
        fontsize='large',
        y=1
    )
    figure.axes[0].set_xlabel('Прогнозируемые значения')
    figure.axes[0].set_ylabel('Фактические значения')
    figure.axes[0].set_xticks(np.linspace(0.0, 0.7, 15))
    figure.axes[0].set_yticks(np.linspace(0.0, 1.0, 11))
    figure.axes[0].set_xlim((0.0, 0.7))
    figure.axes[0].set_ylim((0.0, 1.0))

    rmse = root_mean_squared_error(y_true, y_predict)
    figure.axes[0].legend(
        title='Измерение ошибки модели',
        labels=[f'Ошибка (RMSE: {rmse:.3f})', 'Идеальная модель'],
        loc='upper left',
        alignment='left'
    )

    sns.scatterplot(
        x=y_predict.values,
        y=y_true.values - y_predict.values,
        sizes=8,
        ax=figure.axes[1],
        color=sns.color_palette('hls', 15)[8]
    )

    # Добавляем линию идеальной модели.
    figure.axes[1].axline(
        xy1=(0, 0),
        xy2=(1, 0),
        color='grey',
        linestyle=':',
        label='Идеальная модель'
    )

    # Определяем стиль для графика 2.
    # Определяем заголовок.
    figure.axes[1].set_title(
        label='Остатки и прогнозируемые значения',
        fontsize='large',
        y=1
    )
    figure.axes[1].set_xlabel('Прогнозируемые значения')
    figure.axes[1].set_ylabel('Остатки')
    figure.axes[1].set_xticks(np.linspace(0.0, 0.7, 15))
    figure.axes[1].set_yticks(np.linspace(-0.5, 0.5, 11))
    figure.axes[1].set_xlim((0.0, 0.7))
    figure.axes[1].set_ylim((-0.5, 0.5))

    rmse = root_mean_squared_error(y_true, y_predict)
    figure.axes[1].legend(
        title='Измерение ошибки модели',
        labels=[f'Ошибка (RMSE: {rmse:.3f})', 'Идеальная модель'],
        loc='upper left',
        alignment='left'
    )

    for i in range(2):
        for s in 'top', 'right', 'bottom', 'left':
            figure.axes[i].spines[s].set_visible(False)

    # Сохраняем фигуру в файл.
    if path:
        figure.savefig(
            fname=path + r'\error.png',
            bbox_inches='tight',
            dpi=500
        )
