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
    'estimator__max_depth': np.arange(2, 9, 2).tolist(),
    'estimator__n_estimators': np.arange(250, 1001, 250).tolist(),
    'estimator__max_leaves': np.arange(20, 41, 20).tolist(),
    'estimator__reg_alpha': np.linspace(0.0, 0.75, 4).tolist(),
    'estimator__reg_lambda': np.linspace(0.0, 0.75, 4).tolist(),
}
