import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

from config.ml import RANDOM_STATE


title = 'HistGradientBoostingRegressor'

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
    'estimator__max_iter': [250, 500, 1000],
    'estimator__max_leaf_nodes': np.arange(
        start=20,
        stop=51,
        step=10
    ).tolist(),
    'estimator__min_samples_leaf': np.arange(
        start=20,
        stop=51,
        step=10
    ).tolist(),
    'estimator__l2_regularization': [0.0, 0.25, 0.5, 0.75],
}
