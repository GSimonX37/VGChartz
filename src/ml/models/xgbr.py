import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBRegressor

from config.ml import RANDOM_STATE


title = 'XGBRegressor'

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


estimator = XGBRegressor(
    random_state=RANDOM_STATE,
    verbosity=0
)

model = Pipeline(
    steps=[
        ('standardizer', standardizer),
        ('estimator', estimator),
    ]
)

params = {
    'estimator__learning_rate': [0.1, 0.5, 1],
    'estimator__max_depth': np.arange(
        start=2,
        stop=9,
        step=2
    ).tolist(),
    'estimator__n_estimators': [250, 500, 1000],
    'estimator__max_leaves': np.arange(
        start=20,
        stop=51,
        step=10
    ).tolist(),
    'estimator__reg_alpha': [0.0, 0.25, 0.5, 0.75],
    'estimator__reg_lambda': [0.0, 0.25, 0.5, 0.75]
}
