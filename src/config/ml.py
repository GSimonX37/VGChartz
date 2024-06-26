from numpy import linspace


CV_N_SPLITS = 15
CV_TRAIN_SIZE = 2000
CV_TEST_SIZE = 200
CV_VERBOSE = 3

LEARNING_CURVE_TRAIN_SIZES = [0.01, 0.05]
LEARNING_CURVE_TRAIN_SIZES += [*linspace(0.2, 1., 5)]

N_JOBS = 1
RANDOM_STATE = 42
TEST_SIZE = 200
