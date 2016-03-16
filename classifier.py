from itertools import combinations_with_replacement, islice
import numpy as np
from sklearn.externals import joblib
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer


NUM_BITS_FOR_HASHING = 24
CLASSES = (0.0, 1.0,)


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


def namespace_and_quad_features(X, feature_names):
    def myfunc(row):
        return quad_features(
            prepend_feature_names(
                feature_names,
                row))

    for row in X:
        yield myfunc(row)


def make_pre_processing_pipeline(feature_names, num_bits_for_hashing):
    def namespace_and_quad_features_names_fixed(X):
        return namespace_and_quad_features(X, feature_names)

    namespace_and_quad_feature_transformer = FunctionTransformer(
        namespace_and_quad_features_names_fixed,
        validate=False
    )

    hasher = FeatureHasher(
        input_type='string',
        n_features=2 ** num_bits_for_hashing)

    result = make_pipeline(
        namespace_and_quad_feature_transformer,
        hasher,
        Normalizer()
    )

    return result


def take(n, iterable):
    """Return first n items of the iterable as a list"""
    return list(islice(iterable, n))


def batched_lines(batch_size, parsed_lines):
        batch = take(batch_size, parsed_lines)
        if len(batch) > 0:
            rows = [x for (x, _) in batch]
            labels = np.array([y for (_, y) in batch])
            yield rows, labels


def main():
    # Get training and model filenames
    train_filename = 'train_w_header.txt'
    model_filename = 'model.pkl'

    with open(train_filename) as f:
        lines = (tuple(line.rstrip('\n').split('\t')) for line in f)
        parsed_lines = ((line[1:], float(line[0])) for line in lines)

        # Parse header and get feature names for namespacing
        header = next(lines)
        FEATURE_NAMES = tuple(header[1:])

        # Build pipeline
        pre_processing_pipeline = make_pre_processing_pipeline(
            feature_names=FEATURE_NAMES,
            num_bits_for_hashing=NUM_BITS_FOR_HASHING
        )

        # Instantiate classifier
        # (a logistic regression model with Stochastic Gradient Descent)
        clf = SGDClassifier(loss='log')

        # Train model in mini-batches
        batch_size = 8000

        for rows, labels in batched_lines(batch_size, parsed_lines):
            processed_rows = pre_processing_pipeline.fit_transform(rows)
            clf.partial_fit(processed_rows, labels, classes=CLASSES)

        print clf

        # Save model
        joblib.dump(clf, model_filename)

        # Reload just to make sure it serializes and de- properly
        joblib.load(model_filename)


if __name__ == '__main__':
    main()
