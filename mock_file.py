import tensorflow as tf

FLIPPING_TENSOR = tf.constant([1.0, -1.0, 1.0])

@tf.function
def sample_data(points, labels, num_point):
    if tf.random.uniform(shape=()) >= 0.5:
        return points * FLIPPING_TENSOR, labels

    return points, labels


mock_data = tf.constant([
    [1., 2., 3.],
    [4., 5., 6.],
    [7., 8., 9.]
])

mock_labels = tf.constant([
    [1.], [0.], [1.]
])

sampling_lambda = lambda x, y: sample_data(x, y, 512)

train_data = tf.data.Dataset.from_tensors((mock_data, mock_labels)) \
    .map(sampling_lambda) \
    .unbatch() \
    .batch(1) \
    .repeat(5)

for x, y in train_data:
    print(x)