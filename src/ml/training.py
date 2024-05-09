import csv
import json
import os

from multiprocessing import Pool
from random import randint

import joblib
import optuna
import optuna.logging
import pandas as pd

from optuna.samplers import TPESampler
from optuna.study import Study
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split

from config.ml import CV_TRAIN_SIZE
from config.ml import LEARNING_CURVE_TRAIN_SIZES
from config.ml import N_JOBS
from config.ml import RANDOM_STATE
from config.ml import TEST_SIZE
from config.paths import TRAIN_MODELS_REPORT_PATH
from config.paths import TRAINED_MODELS_PATH
from ml.models.model import Model
from utils import plot
from utils.ml.report import Report
from utils.ml.verbose import Verbose


optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize(n_trials: int,
             model,
             verbose: Verbose,
             report: Report,
             seed: int) -> tuple[Study, Report]:
    """
    Запускает исследование в отдельном процессе;

    :param n_trials: количество испытаний для одного исследования;
    :param model: модель;
    :param verbose: класс для отображения прогресса подбора гиперпараметров;
    :param report: класс для систематизации результатов исследования;
    :param seed: инициализатор TPESampler;
    :return: результаты исследования.
    """

    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=seed)
    )

    study.optimize(
        model,
        n_trials=n_trials,
        callbacks=[verbose, report]
    )

    return study, report


def train(models: dict[str: Model],
          data: pd.DataFrame,
          n_trials: int = 10,
          n_jobs: int = 1) -> None:
    """
    Обучает модели;

    :param models: словарь моделей;
    :param data: набор данных;
    :param n_trials: количество испытаний для каждого исследования;
    :param n_jobs: количество ядер процессора,
    задействованных в подборе гипперпараметров;
    :return: None.
    """

    # Разделение на выборки.
    x = data.drop('total', axis=1)
    y = data['total']

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        shuffle=False,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    for name, model in models.items():
        print(f'{model.name}: {n_jobs*n_trials} [{n_jobs} X {n_trials}].')

        model.x, model.y = x_train, y_train

        args = []
        for i in range(n_jobs):
            verbose = Verbose(n_trials, model.name, i + 1)
            report = Report(i + 1)
            seed = randint(1, 10_000)
            args += [[n_trials, model, verbose, report, seed]]

        with Pool(n_jobs) as pool:
            studies: list[Study, Report] = pool.starmap(optimize, args)

        best_study = max(studies, key=lambda s: s[0].best_value)[0]

        path = fr'{TRAIN_MODELS_REPORT_PATH}\{name}'
        if not os.path.exists(path):
            os.mkdir(path)

        if not os.path.exists(fr'{path}\images'):
            os.mkdir(fr'{path}\images')

        # Сохранение лучших гиперпараметров.
        with open(rf'{path}\params.json', 'w') as f:
            f.write(json.dumps(
                obj=best_study.best_params,
                sort_keys=True,
                indent=4)
            )

        # Сохранение результатов всех испытаний.
        trials = [[
            'job',
            'index',
            'state',
            'start',
            'complete'
        ]]
        trials[-1] += [*model.params.keys()] + ['values', 'best']

        for study in studies:
            trials += study[1].data

        with open(fr'{path}\trials.csv', 'w',
                  newline='',
                  encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows(trials)

        plot.studies(
                trials=pd.DataFrame(trials[1:], columns=trials[0]),
                title=model.name,
                path=fr'{path}\images'
        )

        pipeline = model.pipeline.set_params(**best_study.best_params)
        pipeline.fit(x_train, y_train)

        # Оценка масштабируемости.
        (train_sizes,
         train_scores, test_scores,
         fit_times, score_times) = learning_curve(
            estimator=pipeline,
            X=x_train,
            y=y_train,
            train_sizes=LEARNING_CURVE_TRAIN_SIZES,
            cv=model.cv,
            n_jobs=N_JOBS,
            scoring=model.scoring,
            return_times=True,
            verbose=0
        )

        x_train_size = CV_TRAIN_SIZE
        plot.scalability(
            train_sizes=pd.Series((train_sizes / x_train_size * 100).round(1)),
            train_scores=pd.DataFrame(train_scores),
            test_scores=pd.DataFrame(test_scores),
            fit_times=pd.DataFrame(fit_times),
            score_times=pd.DataFrame(score_times),
            title=f'Масштабируемость {model.name}',
            path=fr'{path}\images'
        )

        # Проверка на тестовой выборке
        predict = pipeline.predict(x_test)
        metric = model.metric(y_test, predict)

        plot.error(
            y_true=pd.Series(y_test),
            y_predict=pd.Series(predict),
            title=f'Ошибки прогнозирования модели {model.name}'
                  f'(RMSE: {metric:.4f}) на тестовой выборке',
            path=fr'{path}\images'
        )

        dummy_rgs = DummyRegressor(strategy="mean")
        dummy_rgs.fit(x_train, y_train)

        predict = dummy_rgs.predict(x_test)
        metric = model.metric(y_test, predict)

        plot.error(
            y_true=pd.Series(y_test),
            y_predict=pd.Series(predict),
            title=f'Результаты обучения простой эмпирической модели'
                  f'(RMSE: {metric:.4f}) на тестовой выборке',
            name='dummy',
            path=fr'{path}\images'
        )

        path = fr'{TRAINED_MODELS_PATH}\{name}'
        if not os.path.exists(path):
            os.mkdir(path)

        # Сохранение модели в joblib-файл.
        joblib.dump(
            value=pipeline,
            filename=rf'{path}\{name}.joblib'
        )
