import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
from sklearn.tree import DecisionTreeRegressor

from config.ml import RANDOM_STATE


title = 'DecisionTreeRegressor'

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


estimator = DecisionTreeRegressor(
        random_state=RANDOM_STATE
)

model = Pipeline(
    steps=[
        ('standardizer', standardizer),
        ('estimator', estimator),
    ]
)

params = {
    'estimator__max_depth': np.arange(
        start=2,
        stop=21
    ).tolist(),
    'estimator__min_samples_split': np.arange(
        start=2,
        stop=21,
        step=2
    ).tolist(),
}
