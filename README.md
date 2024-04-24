def parser_images(location, label):
    # Your implementation to load and return images and labels
    # Example:
    image = load_image(location)
    return image, label


def preprocess_img(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [desired_height, desired_width])
    return image, label
# Assume train_locations and train_labels are lists of image locations and labels
train_dataset = tf.data.Dataset.from_tensor_slices((train_locations, train_labels))

# Use parser_images to parse the images and labels
train_dataset = train_dataset.map(parser_images)

# Use preprocess_img to preprocess the images
train_dataset = train_dataset.map(preprocess_img)

# Shuffle and batch the dataset
train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size).prefetch(buffer_size)
#specify batch and train_size on own

# Naive Bayes algorithm types
## Multinomial Naive Bayes:
Suitable for text classification tasks where features represent word counts or term frequencies.
Commonly used in natural language processing (NLP) tasks such as document classification, spam filtering, sentiment analysis, etc.
Works well with data represented as sparse matrices (e.g., TF-IDF vectors).
## Gaussian Naive Bayes:
Suitable for continuous data where features follow a Gaussian (normal) distribution.
Works well for data with numerical features that are assumed to be normally distributed.
Commonly used in classification tasks involving numerical data, such as medical diagnosis, financial analysis, etc.
## Bernoulli Naive Bayes:
Suitable for binary and categorical data represented as binary features (0/1 or True/False).
Often used in text classification tasks with binary features (e.g., presence or absence of words in a document).
Works well when dealing with data that can be modeled as a sequence of binary decisions.
