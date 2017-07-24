import os
import sys
import errno

import cv2
import numpy as np

def grayscale(image):
    assert image.dtype == np.uint8
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = image[:, :, 2]
    return image

def preprocess(image, model_input_size):
    rows, columns = model_input_size
    resized_image = cv2.resize(image, (columns, rows))
    return grayscale(resized_image)

def pair_difference(a, b):
    assert a.dtype == b.dtype == np.uint8
    # Rescale a - b from [-255, 255] to [0, 255]
    rescaled = (a.astype(np.int16) - b.astype(np.int16) + 255) / 2
    return rescaled.astype(np.uint8)

def image_sequence_preprocessing(images, sequence_length, model_input_size = (192, 256)):
    last_images = images[-sequence_length - 1:]
    assert len(last_images) == sequence_length + 1

    preprocessed = [preprocess(image, model_input_size) for image in last_images]
    image_differences = [
        pair_difference(a, b)
        for a, b in zip(preprocessed[:-1], preprocessed[1:])
    ]
    return image_differences

def batch_preprocessing(
    pair_first_list_name,
    pair_second_list_name,
    output_directory,
    output_template = "{:07d}.jpeg",
    model_input_size = (192, 256)
):
    for index, (pair_first_name, pair_second_name) in enumerate(zip(
        open(pair_first_list_name),
        open(pair_second_list_name),
    )):
        pair_first_name, pair_second_name = [
            image_name.rstrip('\n')
            for image_name in (pair_first_name, pair_second_name)
        ]
        assert all([os.path.exists(image_name) for image_name in (pair_first_name, pair_second_name)])
        images_pair = [
            cv2.imread(image_name) for image_name in (pair_first_name, pair_second_name)
        ]

        assert all([image is not None for image in images_pair])
        preprocessed = image_sequence_preprocessing(images_pair, 1, model_input_size)[0]
        cv2.imwrite(os.path.join(output_directory, output_template.format(index)), preprocessed)
        if index % 1000 == 0:
            print index

if __name__ == '__main__':
    first_list_name = sys.argv[1]
    second_list_name = sys.argv[2]
    output_directory = sys.argv[3]
    try:
        os.mkdir(output_directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        print >>sys.stderr, "Directory %s is already exists" % output_directory

    batch_preprocessing(first_list_name, second_list_name, output_directory)