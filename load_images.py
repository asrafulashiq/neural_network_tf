import tensorflow as tf
import os

test_folder = 'test_data'
train_folder = 'train_data'

test_label = 'labels/test_label.txt'
train_label = 'labels/train_label.txt'

def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=1)
    image = tf.cast(image_decoded, tf.float32)
    image = tf.reshape(image, [-1]) / 255.
    label = tf.one_hot(label, 10)
    return image, label

def create_data(folder, label_txt):
    # get all image filenames from folder
    filenames = [os.path.join(folder, i) for i in os.listdir(folder) if i.endswith('jpg')]
    labels = [int(i.strip()) for i in open(label_txt)]

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function).repeat()
    return dataset

if __name__=='__main__':
    dataset = create_data(test_folder, test_label)

    dataset = dataset.batch(2)

    # step 3: create iterator and final input tensor
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()

