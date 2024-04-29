from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.model_selection import TimeSeriesSplit

from .model import Model
from config.ml import RANDOM_STATE
from config.ml import CV_N_SPLITS
from config.ml import CV_TRAIN_SIZE
from config.ml import CV_TEST_SIZE


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


estimator = LGBMRegressor(
    random_state=RANDOM_STATE,
    verbosity=-1
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
    'estimator__n_estimators': ['int', {'low': 50,
                                        'high': 500,
                                        'step': 50}],
    'estimator__num_leaves': ['int', {'low': 2,
                                      'high': 50,
                                      'step': 2}],
    'estimator__reg_alpha': ['float', {'low': 0.0,
                                       'high': 1.0,
                                       'step': 0.05}],
    'estimator__reg_lambda': ['float', {'low': 0.0,
                                        'high': 1.0,
                                        'step': 0.05}],
}

scoring = make_scorer(
    score_func=mean_squared_error,
    zero_division=0.0
)

cv = TimeSeriesSplit(
    n_splits=CV_N_SPLITS,
    max_train_size=CV_TRAIN_SIZE,
    test_size=CV_TEST_SIZE
)


pipeline = Model(
    pipeline=pipeline,
    name='LGBMRegressor',
    params=params,
    metric=mean_squared_error,
    scoring=scoring,
    cv=cv
)
