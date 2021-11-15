from typing import Tuple
import tensorflow as tf


def get_binary_kernel2d(window_size: Tuple[int, int]) -> tf.Tensor:
    for i in range(window_size[0]):
        for j in range(window_size[1]):
            x = tf.sparse_tensor_to_dense(tf.SparseTensor(indices=[[i, j]], values=[1.0], dense_shape=[3, 3]))
            x1 = tf.expand_dims(tf.expand_dims(x, 0), 0)
            if i == 0 and j == 0:
                y = tf.concat([x1], 0)
            else:
                y = tf.concat([y, x1], 0)

    return y  # kernels (9,1,3,3)


def get_median(input_g: tf.Tensor, dim: int):  # [b, c, area, h, w]
    area = tf.shape(input_g)[dim]  # 几个数的中值
    patches = tf.transpose(input_g, [0, 3, 4, 1, 2])
    floor = (area + 1) // 2
    ceil = area // 2 + 1
    top = tf.nn.top_k(patches, k=ceil).values
    if area % 2 == 1:
        median = top[:, :, :, :, floor - 1]
    else:
        median = (top[:, :, :, :, floor - 1] + top[:, :, :, :, ceil - 1]) / 2

    return median


def median_blur(input_m: tf.Tensor, kernel_size: Tuple[int, int]) -> tf.Tensor:
    if not isinstance(input_m, tf.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input_m)}")

    if not len(input_m.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input_m.shape}")

    # prepare kernel
    kernel = get_binary_kernel2d(kernel_size)
    b, c, h, w = input_m.shape
    features = tf.nn.conv2d(tf.transpose(tf.reshape(input_m, [b * c, 1, h, w]), [0, 2, 3, 1]),
                            tf.transpose(kernel, [2, 3, 1, 0]), padding="SAME",
                            strides=[1, 1, 1, 1])
    features = tf.reshape(tf.transpose(features, [0, 3, 1, 2]), [b, c, -1, h, w])  # BxCx(K_h * K_w)xHxW

    # compute the median along the feature axis
    median = get_median(features, dim=2)
    median = tf.transpose(median, [0, 3, 1, 2])

    return median


def mean_blur(input_m: tf.Tensor, kernel_size: Tuple[int, int]) -> tf.Tensor:
    if not isinstance(input_m, tf.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input_m)}")

    if not len(input_m.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input_m.shape}")

    # prepare kernel
    kernel = get_binary_kernel2d(kernel_size)
    b, c, h, w = input_m.shape
    features = tf.nn.conv2d(tf.transpose(tf.reshape(input_m, [b * c, 1, h, w]), [0, 2, 3, 1]),
                            tf.transpose(kernel, [2, 3, 1, 0]), padding="SAME",
                            strides=[1, 1, 1, 1])
    features = tf.reshape(tf.transpose(features, [0, 3, 1, 2]), [b, c, -1, h, w])  # BxCx(K_h * K_w)xHxW

    # compute the median along the feature axis
    mean = tf.reduce_mean(features, axis=2)  # math.

    return mean
