from numpy import linspace


FIT_CV_VERBOSE = 3
FIT_CV_N_SPLITS = 3
FIT_CV_TEST_SIZE = 500
LEARNING_CURVE_N_SPLITS = 5
LEARNING_CURVE_TEST_SIZE = 500
LEARNING_CURVE_TRAIN_SIZES = [0.01, 0.05]
LEARNING_CURVE_TRAIN_SIZES += [*linspace(0.2, 1., 5)]
LEARNING_CURVE_VERBOSE = 3
N_JOBS = 4
RANDOM_STATE = 42
TEST_SIZE = 0.1
