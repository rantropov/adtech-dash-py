from itertools import combinations_with_replacement
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer


NUM_BITS_FOR_HASHING = 24
FEATURE_NAMES = tuple(
    ['i' + str(i) for i in xrange(1, 14)] +
    ['c' + str(i) for i in xrange(1, 27)])


def prepend_feature_names(feature_names, row):
    if len(feature_names) != len(row):
        raise IndexError
    return map(lambda x: ''.join(x), zip(feature_names, row))


def quad_features(row, delimiter=':'):
    return map(
        lambda (x, y): x + delimiter + y,
        combinations_with_replacement(
            row,
            2))


def namespace_and_quad_features(X):
    def myfunc(row):
        return quad_features(
            prepend_feature_names(
                FEATURE_NAMES,
                row))
    for row in X:
        yield myfunc(row)


def main():
    with open('dac_sample.txt') as f:
        lines = [line.rstrip('\n').split('\t')[1:] for line in f]

        namespace_and_quad_feature_transformer = FunctionTransformer(
            namespace_and_quad_features,
            validate=False
        )

        hasher = FeatureHasher(
            input_type='string',
            n_features=2 ** NUM_BITS_FOR_HASHING)

        preprocessing_pipeline = make_pipeline(
            namespace_and_quad_feature_transformer,
            hasher,
            Normalizer()
        )

        print preprocessing_pipeline.fit_transform(lines[:10000]).shape
    print preprocessing_pipeline


if __name__ == '__main__':
    main()