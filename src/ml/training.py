import csv
import json
import os

import joblib
import optuna
import optuna.logging
import pandas as pd

from optuna.samplers import TPESampler
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
from utils.ml.verbose import Verbose


optuna.logging.set_verbosity(optuna.logging.WARNING)


def train(models: dict[str: Model],
          data: pd.DataFrame,
          n_trials: int = 10,
          n_jobs: int = 1) -> None:
    """
    Обучает модели;

    :param models: словарь моделей;
    :param data: набор данных;
    :param n_trials: количество испытаний;
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
        print(f'{model.name}: {n_trials}.')

        model.x, model.y = x_train, y_train

        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler()
        )

        verbose = Verbose(n_trials, model.name)

        study.optimize(
            model,
            n_trials=n_trials,
            n_jobs=n_jobs,
            callbacks=[verbose]
        )

        path = fr'{TRAIN_MODELS_REPORT_PATH}\{name}'
        if not os.path.exists(path):
            os.mkdir(path)

        if not os.path.exists(fr'{path}\images'):
            os.mkdir(fr'{path}\images')

            # Сохранение лучших гиперпараметров.
            with open(rf'{path}\params.json', 'w') as f:
                f.write(json.dumps(
                    obj=study.best_params,
                    sort_keys=True,
                    indent=4)
                )

        # Сохранение результатов всех испытаний.
        trials = [[
            'index',
            'state',
            'start',
            'complete'
        ]]
        trials += [model.params.keys()] + ['values']
        for trial in study.trials:
            index = trial.number + 1
            state = trial.state.name
            start = (trial
                     .datetime_start
                     .strftime('%d-%m-%Y %H:%M:%S'))
            complete = (trial
                        .datetime_complete
                        .strftime('%d-%m-%Y %H:%M:%S'))
            params = trial.params.values()
            params = [round(p, 4) if isinstance(p, (int, float)) else p
                      for p in params]
            value = round(trial.values[0], 4)

            trials.append([
                index,
                state,
                start,
                complete
            ])

            trials[-1] += [*params] + [value]

        with open(fr'{path}\trials.csv', 'w',
                  newline='',
                  encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows(trials)

        pipeline = model.pipeline.set_params(**study.best_params)
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
