# Assuming we have already run the Doc2Vec model and the necessary variables are available.

# Maximum number of words in a document
max_words = 20

# Logistic regression batch size
logistic_batch_size = 500

# Split dataset into train and test sets
# Need to keep the indices sorted to keep track of the document index
train_indices = np.sort(np.random.choice(len(target), round(0.8*len(target)), replace=False))
test_indices = np.sort(np.array(list(set(range(len(target))) - set(train_indices))))
texts_train = [x for ix, x in enumerate(texts) if ix in train_indices]
texts_test = [x for ix, x in enumerate(texts) if ix in test_indices]
target_train = np.array([x for ix, x in enumerate(target) if ix in train_indices])
target_test = np.array([x for ix, x in enumerate(target) if ix in test_indices])

# Convert texts to lists of indices
text_data_train = np.array(text_helpers.text_to_numbers(texts_train, word_dictionary))
text_data_test = np.array(text_helpers.text_to_numbers(texts_test, word_dictionary))

# Pad/crop movie reviews to a specific length
text_data_train = np.array([x[0:max_words] for x in [y+[0]*max_words for y in text_data_train]])
text_data_test = np.array([x[0:max_words] for x in [y+[0]*max_words for y in text_data_test]])

# Create variables for logistic regression
log_x_inputs = tf.Variable(tf.zeros([logistic_batch_size, max_words + 1], dtype=tf.int32))  # plus 1 for doc index
log_y_target = tf.Variable(tf.zeros([logistic_batch_size, 1], dtype=tf.int32))

# Define logistic embedding lookup (needed if we have two different batch sizes)
# Add together element embeddings in the window:
log_embed = tf.zeros([logistic_batch_size, embedding_size])
for element in range(max_words):
    log_embed += tf.nn.embedding_lookup(embeddings, log_x_inputs[:, element])

log_doc_indices = tf.slice(log_x_inputs, [0, max_words], [logistic_batch_size, 1])
log_doc_embed = tf.nn.embedding_lookup(doc_embeddings, log_doc_indices)

# concatenate embeddings
log_final_embed = tf.concat(axis=1, values=[log_embed, tf.squeeze(log_doc_embed)])

# Define model:
# Create variables for logistic regression
A = tf.Variable(tf.random.normal(shape=[concatenated_size, 1]))
b = tf.Variable(tf.random.normal(shape=[1, 1]))


# Define logistic model (sigmoid in loss function)

concatenated_size = 300

# Inside the logistic_model function
@tf.function
def logistic_model(log_final_embed):
    log_final_embed_int64 = tf.cast(log_final_embed, tf.float32)

    # Assuming log_final_embed_int64 has shape [batch_size, concatenated_size]
    # If the size is different, adjust it accordingly
    assert log_final_embed_int64.shape[1] == concatenated_size, "Incorrect dimensions for log_final_embed"

    # Adjust A to have dimensions [concatenated_size, 1]
    A_adjusted = tf.transpose(A)

    # Change the following line to perform matrix multiplication correctly
    return tf.add(tf.matmul(log_final_embed_int64, A_adjusted), b)

# Define loss function (Cross Entropy loss)
@tf.function
def logistic_loss_fn(log_final_embed, log_y_target):
    logits = logistic_model(log_final_embed)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.cast(log_y_target, tf.float32)))

# Actual Prediction
@tf.function
def logistic_prediction(log_final_embed):
    return tf.round(tf.sigmoid(logistic_model(log_final_embed)))

# Compute gradients
@tf.function
def compute_gradients():
    with tf.GradientTape() as tape:
        current_loss = logistic_loss_fn(log_final_embed, log_y_target)

    grads = tape.gradient(current_loss, [A, b])
    return grads, current_loss

# Declare optimizer
logistic_opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)

# Start Logistic Regression
print('Starting Logistic Doc2Vec Model Training')
train_loss = []
test_loss = []
train_acc = []
test_acc = []
i_data = []
for i in range(10000):
    rand_index = np.random.choice(text_data_train.shape[0], size=logistic_batch_size)
    rand_x = text_data_train[rand_index]
    # Append review index at the end of text data
    rand_x_doc_indices = train_indices[rand_index]
    rand_x = np.hstack((rand_x, np.transpose([rand_x_doc_indices])))
    rand_y = np.transpose([target_train[rand_index]])

    grads, current_loss = compute_gradients()
    logistic_opt.apply_gradients(zip(grads, [A, b]))

    # Only record loss and accuracy every 100 generations
    if (i + 1) % 100 == 0:
        rand_index_test = np.random.choice(text_data_test.shape[0], size=logistic_batch_size)
        rand_x_test = text_data_test[rand_index_test]
        # Append review index at the end of text data
        rand_x_doc_indices_test = test_indices[rand_index_test]
        rand_x_test = np.hstack((rand_x_test, np.transpose([rand_x_doc_indices_test])))
        rand_y_test = np.transpose([target_test[rand_index_test]])

        test_loss_temp = logistic_loss_fn(log_final_embed, rand_y_test)
        train_acc_temp = tf.reduce_mean(tf.cast(tf.equal(logistic_prediction(rand_x), tf.cast(rand_y, tf.float32)), tf.float32))
        test_acc_temp = tf.reduce_mean(tf.cast(tf.equal(logistic_prediction(rand_x_test), tf.cast(rand_y_test, tf.float32)), tf.float32))

        i_data.append(i + 1)
        train_loss.append(current_loss)
        test_loss.append(test_loss_temp)
        train_acc.append(train_acc_temp)
        test_acc.append(test_acc_temp)

    if (i + 1) % 500 == 0:
        acc_and_loss = [i + 1, current_loss, test_loss_temp, train_acc_temp, test_acc_temp]
        acc_and_loss = [np.round(x, 2) for x in acc_and_loss]
        print('Generation # {}. Train Loss (Test Loss): {:.2f} ({:.2f}). Train Acc (Test Acc): {:.2f} ({:.2f})'.format(
            *acc_and_loss))

# Plot loss over time
plt.plot(i_data, train_loss, 'k-', label='Train Loss')
plt.plot(i_data, test_loss, 'r--', label='Test Loss', linewidth=4)
plt.title('Cross Entropy Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Cross Entropy Loss')
plt.legend(loc='upper right')
plt.show()

# Plot train and test accuracy
plt.plot(i_data, train_acc, 'k-', label='Train Set Accuracy')
plt.plot(i_data, test_acc, 'r--', label='Test Set Accuracy', linewidth=4)
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()