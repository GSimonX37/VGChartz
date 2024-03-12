import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

from config.ml import RANDOM_STATE


title = 'RandomForestRegressor'

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


estimator = RandomForestRegressor(
    random_state=RANDOM_STATE
)

model = Pipeline(
    steps=[
        ('standardizer', standardizer),
        ('estimator', estimator),
    ]
)

params = {
    'estimator__n_estimators': np.arange(150, 451, 150),
    'estimator__max_depth': np.arange(4, 13, 4),
    'estimator__min_samples_split': np.arange(2, 11, 4),
    'estimator__min_samples_leaf': np.arange(1, 10, 4)
}
