import json
import os

import joblib
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split

from config.ml import CV_N_SPLITS
from config.ml import CV_TEST_SIZE
from config.ml import CV_TRAIN_SIZE
from config.ml import CV_VERBOSE
from config.ml import LEARNING_CURVE_TRAIN_SIZES
from config.ml import N_JOBS
from config.ml import RANDOM_STATE
from config.ml import TEST_SIZE
from config.paths import FILE_PREPROCESSED_PATH
from config.paths import TRAIN_MODELS_REPORT_PATH
from config.paths import TRAINED_MODELS_PATH
from utils.ml.plot.error import error
from utils.ml.plot.scalability import scalability


def train(file: str, models: list) -> None:
    """
    Обучает модели;

    :param file: имя предобработанного файла в формате csv;
    :param models: список с параметрами тренируемых моделей;
    :return: None.
    """

    df = pd.read_csv(fr'{FILE_PREPROCESSED_PATH}\{file}')

    # Изменение типов данных.
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df = df.astype({
        'shipped': 'float32',
        'total': 'float32',
        'america': 'float32',
        'europe': 'float32',
        'japan': 'float32',
        'other': 'float32',
        'vgc': 'float32',
        'critic': 'float32',
        'user': 'float32',
    })

    # Отбор данных.
    columns = ['date', 'platform', 'publisher', 'developer', 'total']
    data = df[(df[columns].notna().all(axis=1)) & (df['total'] < 1.0)]

    countries = ['america', 'europe', 'japan', 'other']
    data = data[columns + countries]

    data = data[data['total'] != 0.0]
    data.loc[:, countries] = data.loc[:, countries].fillna(0)
    data.loc[:, countries] = data.loc[:, countries] != 0.0
    data = data.sort_values(by='date').reset_index(drop=True)
    data = data.drop(columns=['date'])

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

    cv = TimeSeriesSplit(
        n_splits=CV_N_SPLITS,
        max_train_size=CV_TRAIN_SIZE,
        test_size=CV_TEST_SIZE
    )

    scoring = 'neg_root_mean_squared_error'

    for model in models:
        name, title, model, params = (
            model['name'],
            model['title'],
            model['model'],
            model['params'],
        )

        path = fr'{TRAIN_MODELS_REPORT_PATH}\{name}'
        if not os.path.exists(path):
            os.mkdir(path)

        if not os.path.exists(fr'{path}\images'):
            os.mkdir(fr'{path}\images')

        # Обучение модели.
        clf = GridSearchCV(
            estimator=model,
            param_grid=params,
            scoring=scoring,
            cv=cv,
            verbose=CV_VERBOSE,
            refit=True
        )
        clf.fit(x_train, y_train)

        # Сохранение результатов кросс-валидации.
        (pd.DataFrame(data=clf.cv_results_)
         .sort_values('rank_test_score')
         .round(5)
         .to_csv(
            path_or_buf=rf'{path}\cv_results.csv',
            sep=',',
            index=False
        ))

        # Сохранение лучших гиперпараметров.
        with open(rf'{path}\best_params.json', 'w') as f:
            f.write(json.dumps(clf.best_params_, sort_keys=True, indent=4))

        # Оценка масштабируемости.
        (train_sizes,
         train_scores, test_scores,
         fit_times, score_times) = learning_curve(
            estimator=clf.best_estimator_,
            X=x_train,
            y=y_train,
            train_sizes=LEARNING_CURVE_TRAIN_SIZES,
            cv=cv,
            n_jobs=N_JOBS,
            scoring='neg_root_mean_squared_error',
            return_times=True,
            verbose=CV_VERBOSE
        )

        x_train_size = CV_TRAIN_SIZE
        scalability(
            train_sizes=pd.Series((train_sizes / x_train_size * 100).round(1)),
            train_scores=pd.DataFrame(train_scores),
            test_scores=pd.DataFrame(test_scores),
            fit_times=pd.DataFrame(fit_times),
            score_times=pd.DataFrame(score_times),
            title=f'Масштабируемость {title}',
            path=fr'{path}\images'
        )

        # Проверка на тестовой выборке
        predict = clf.predict(x_test)

        error(
            y_true=pd.Series(y_test),
            y_predict=pd.Series(predict),
            title=f'Ошибки прогнозирования регрессионной модели {title}',
            path=fr'{path}\images'
        )

        path = fr'{TRAINED_MODELS_PATH}\{name}'
        if not os.path.exists(path):
            os.mkdir(path)

        model = clf.best_estimator_

        # Сохранение модели в joblib-файл.
        joblib.dump(
            value=model,
            filename=rf'{path}\{name}.joblib'
        )
