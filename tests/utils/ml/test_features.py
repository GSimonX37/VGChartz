import numpy as np
import pandas as pd
import pytest

from src.utils.ml.features import countries
from src.utils.ml.features import generate
from src.utils.ml.features import same


columns = ['publisher', 'developer', 'america', 'europe', 'japan', 'other']


@pytest.mark.parametrize(
    ['data', 'expected'],
    [
        (pd.DataFrame(data=[[False, False, False, False]], index=[0]), [0]),
        (pd.DataFrame(data=[[True, False, False, False]], index=[0]), [1]),
        (pd.DataFrame(data=[[False, False, False, True]], index=[0]), [1]),
        (pd.DataFrame(data=[[True, True, True, True]], index=[0]), [4])
    ]
)
def test_countries(data, expected):
    result = countries(data).values.tolist() == expected
    assert result


@pytest.mark.parametrize(
    ['data', 'expected'],
    [
        (pd.DataFrame(
            data=[['Sony', 'Sony']],
            index=[0],
            columns=columns[:2]
        ), [True]),
        (pd.DataFrame(
            data=[['Nintendo', 'Nintendo EAD']],
            index=[0],
            columns=columns[:2]
        ), [False]),
        (pd.DataFrame(
                data=[['Nintendo', 'Nintendo']],
                index=[0],
                columns=columns[:2]
        ), [True]),
        (pd.DataFrame(
                data=[['Spike', 'Bethesda Softworks']],
                index=[0],
                columns=columns[:2]
        ), [False])
    ]
)
def test_same(data, expected):
    result = same(data).values.tolist() == expected
    assert result


@pytest.mark.parametrize(
    ['data'],
    [
        (pd.DataFrame(
                data=[['Sony', 'Sony', False, False, False, False]],
                index=[0],
                columns=columns
        ), ),
        (pd.DataFrame(
                data=[['Nintendo', 'Nintendo EAD', True, False, False, False]],
                index=[0],
                columns=columns
        ), ),
        (pd.DataFrame(
                data=[['Nintendo', 'Nintendo', False, False, False, True]],
                index=[0],
                columns=columns
        ), ),
        (pd.DataFrame(
                data=[['Spike', 'Bethesda Softworks', True, True, True, True]],
                index=[0],
                columns=columns
        ), )
    ]
)
def test_features_data_type(data):
    result = isinstance(generate(data), pd.DataFrame)
    assert result


@pytest.mark.parametrize(
    ['data'],
    [
        (pd.DataFrame(
                data=[['Sony', 'Sony', False, False, False, False]],
                index=[0],
                columns=columns
        ), ),
        (pd.DataFrame(
                data=[['Nintendo', 'Nintendo EAD', True, False, False, False]],
                index=[0],
                columns=columns
        ), ),
        (pd.DataFrame(
                data=[['Nintendo', 'Nintendo', False, False, False, True]],
                index=[0],
                columns=columns
        ), ),
        (pd.DataFrame(
                data=[['Spike', 'Bethesda Softworks', True, True, True, True]],
                index=[0],
                columns=columns
        ), )
    ]
)
def test_features_data_types(data):
    types = generate(data).dtypes.values

    bool_type = types[0] == np.dtypes.BoolDType
    integer_type = types[1] == np.dtypes.Int64DType

    result = bool_type and integer_type

    assert result
