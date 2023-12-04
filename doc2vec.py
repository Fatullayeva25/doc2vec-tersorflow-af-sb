import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Doc2VecModel:
    def __init__(self, vocab_size, embed_size, num_docs, max_sequence_length):
        self._embed_size = embed_size
        self._num_docs = num_docs
        self._max_sequence_length = max_sequence_length
        self._syn0_w, self._syn0_d, self._syn1, self._biases = self._create_embeddings(vocab_size)

    def _create_embeddings(self, vocab_size):
        syn0_w = tf.Variable(tf.random.uniform([vocab_size, self._embed_size], -0.5, 0.5))
        syn0_d = tf.Variable(tf.random.uniform([self._num_docs, self._embed_size], -0.5, 0.5))
        syn1 = tf.Variable(tf.random.uniform([vocab_size, self._embed_size], -0.1, 0.1))
        biases = tf.Variable(tf.zeros([vocab_size]))
        return syn0_w, syn0_d, syn1, biases

    def _negative_sampling_loss(self, unigram_counts, inputs, labels):
        return tf.reduce_mean(tf.nn.sampled_softmax_loss(
            weights=tf.transpose(self._syn1),
            biases=self._biases,
            labels=tf.expand_dims(labels, -1),
            inputs=inputs,
            num_sampled=5,
            num_classes=tf.shape(self._syn1)[0],
            num_true=1
        ))

    def _train_step(self, inputs, labels):
        inputs = tf.clip_by_value(inputs, 0, self._num_docs - 1)
        labels = tf.clip_by_value(labels, 0, self._syn1.shape[0] - 1)

        inputs_syn0 = tf.gather(self._syn0_d, inputs)
        loss = self._custom_loss(inputs_syn0, labels)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(loss)

        return loss, train_op

    def train(self, dataset, num_epochs=10):
        for epoch in range(num_epochs):
            for inputs, labels in dataset:
                inputs = tf.clip_by_value(inputs, 0, self._num_docs - 1)
                loss, train_op = self._train_step(inputs, labels)
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    _, current_loss = sess.run([train_op, loss])
                    print(f'Epoch {epoch + 1}, Loss: {current_loss}')

    def _custom_loss(self, inputs_syn0, labels):
        labels = tf.cast(tf.expand_dims(labels, axis=-1), dtype=tf.float32)

        mask = tf.reduce_all(tf.not_equal(labels, 0), axis=-1)
        inputs_syn0 = tf.boolean_mask(inputs_syn0, mask)
        labels = tf.boolean_mask(labels, mask)

        inputs_syn0 = tf.reshape(inputs_syn0, [-1, tf.shape(labels)[-1]])
        loss = tf.reduce_mean(tf.square(inputs_syn0 - labels))

        return loss


def preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Combine the last 50 posts into a single string
    df['CombinedPosts'] = df.apply(lambda row: '|||'.join(row[-50:]), axis=1)

    # Tokenize posts by '|||' to create a list of posts for each user
    posts = df['CombinedPosts'].apply(lambda x: x.split('|||')).tolist()

    # Flatten the list of lists
    flattened_posts = [post for sublist in posts for post in sublist]

    # Create a vocabulary
    vocab = list(set(flattened_posts))
    vocab_size = len(vocab)

    # Create a mapping from word to index
    word_to_index = {word: i for i, word in enumerate(vocab)}

    # Encode MBTI types
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df['type'])

    # Map each post to its corresponding index in the vocabulary
    posts_indices = [[word_to_index[word] for word in post.split('|||')] for post in df['CombinedPosts']]

    # Pad sequences to have the same length
    max_sequence_length = max(len(post) for post in posts_indices)
    posts_indices_padded = pad_sequences(posts_indices, maxlen=max_sequence_length, padding='post', truncating='post')

    return posts_indices_padded, labels, vocab_size, max_sequence_length

# Path to the "mbti_1.csv" file
file_path = "mbti_1.csv"

# Preprocess the data
posts_indices, labels, vocab_size, max_sequence_length = preprocess_data(file_path)

# Split the data into training and validation sets
train_posts, val_posts, train_labels, val_labels = train_test_split(posts_indices, labels, test_size=0.2, random_state=42)

# Create a TensorFlow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_posts, train_labels)).shuffle(buffer_size=len(train_posts)).batch(32)

# Create an instance of the Doc2VecModel
doc2vec_model = Doc2VecModel(vocab_size, embed_size=100, num_docs=len(train_posts), max_sequence_length=max_sequence_length)

# Train the model
doc2vec_model.train(train_dataset)
