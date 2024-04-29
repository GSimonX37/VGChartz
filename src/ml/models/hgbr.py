from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import make_scorer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

from config.ml import CV_N_SPLITS
from config.ml import CV_TEST_SIZE
from config.ml import CV_TRAIN_SIZE
from config.ml import RANDOM_STATE

from .model import Model


category = [
    'platform',
    'publisher',
    'developer',
    'america',
    'europe',
    'japan',
    'other'
]

encoder = TargetEncoder(
    random_state=RANDOM_STATE
)

standardizer = ColumnTransformer(
    transformers=[
        ('encoder', encoder, category)
    ],
    remainder='passthrough'
)


estimator = HistGradientBoostingRegressor(
    random_state=RANDOM_STATE
)

pipeline = Pipeline(
    steps=[
        ('standardizer', standardizer),
        ('estimator', estimator),
    ]
)

params = {
    'estimator__learning_rate': ['float', {'low': 0.1,
                                           'high': 1.0,
                                           'step': 0.05}],
    'estimator__max_depth': ['int', {'low': 1,
                                     'high': 50,
                                     'step': 1}],
    'estimator__max_iter': ['int', {'low': 50,
                                    'high': 500,
                                    'step': 50}],
    'estimator__max_leaf_nodes': ['int', {'low': 2,
                                          'high': 50,
                                          'step': 2}],
    'estimator__min_samples_leaf': ['int', {'low': 2,
                                            'high': 50,
                                            'step': 2}],
    'estimator__l2_regularization': ['float', {'low': 0.0,
                                               'high': 1.0,
                                               'step': 0.05}],
}

scoring = make_scorer(
    score_func=root_mean_squared_error
)

cv = TimeSeriesSplit(
    n_splits=CV_N_SPLITS,
    max_train_size=CV_TRAIN_SIZE,
    test_size=CV_TEST_SIZE
)


model = Model(
    pipeline=pipeline,
    name='HistGradientBoostingRegressor',
    params=params,
    metric=root_mean_squared_error,
    scoring=scoring,
    cv=cv
)
