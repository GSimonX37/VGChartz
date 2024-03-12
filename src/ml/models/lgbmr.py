import numpy as np

from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

from config.ml import RANDOM_STATE


title = 'LGBMRegressor'

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

model = Pipeline(
    steps=[
        ('standardizer', standardizer),
        ('estimator', estimator),
    ]
)

params = {
    'estimator__learning_rate': [0.1, 0.5, 1],
    'estimator__max_depth': np.arange(2, 9, 2),
    'estimator__n_estimators': np.arange(250, 1001, 250),
    'estimator__num_leaves': np.arange(20, 41, 20),
    'estimator__reg_alpha': np.linspace(0.0, 0.75, 4),
    'estimator__reg_lambda': np.linspace(0.0, 0.75, 4),
}
