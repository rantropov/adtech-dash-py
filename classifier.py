from itertools import combinations_with_replacement, islice
import numpy as np
from sklearn.externals import joblib
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


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


def main():
    with open('dac_sample.txt') as f:
        lines = (tuple(line.rstrip('\n').split('\t')) for line in f)

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

        def first_n(n):
            for _ in xrange(n):
                line = next(lines)
                y = float(line[0])
                x = line[1:]
                yield x, y

        first_30k = first_n(30000)

        clf = SGDClassifier(loss='log')

        batch_size = 8000

        while True:
            batch = take(batch_size, first_30k)
            if len(batch) > 0:
                X = [x for (x, _) in batch]
                Y = np.array([y for (_, y) in batch])
                processed_x = preprocessing_pipeline.fit_transform(X)
                clf.partial_fit(processed_x, Y, classes=[0.0, 1.0])
            else:
                break

        print clf

        joblib.dump(clf, 'model.pkl')
        joblib.load('model.pkl')

if __name__ == '__main__':
    main()
