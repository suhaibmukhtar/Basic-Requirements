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
