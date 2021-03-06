"""Implementation of sample attack."""
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from attack_method import *
from tqdm import tqdm
from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2
import os
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
slim = tf.contrib.slim

tf.flags.DEFINE_string('checkpoint_path', '../../0models', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string('input_csv', '../../0dataset/dev_dataset.csv', 'Input directory with images.')

tf.flags.DEFINE_string('input_dir', '../../0dataset/images/', 'Input directory with images.')

tf.flags.DEFINE_string('output_dir', 'output/', 'Output directory with images.')

tf.flags.DEFINE_float('max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer('num_classes', 1001, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer('num_iter', 1, 'Number of iterations.')  # n=1

tf.flags.DEFINE_integer('image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer('image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer('image_resize', 330, 'Height of each input images.')

tf.flags.DEFINE_integer('batch_size', 10, 'How many images process at one time.')

tf.flags.DEFINE_float('amplification_factor', 10.0, 'To amplifythe step size.')

tf.flags.DEFINE_float('momentum', 1.0, 'Momentum.')

tf.flags.DEFINE_float('prob', 0.7, 'probability of using diverse inputs.')

tf.flags.DEFINE_string('train_model', 'Inc-v3', 'Inc-v3/Inc-v4/Res152/IncRes.')

FLAGS = tf.flags.FLAGS

model_checkpoint_map = {
    'inception_v3': os.path.join(FLAGS.checkpoint_path, 'inception_v3.ckpt'),
    'adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'adv_inception_v3_rename.ckpt'),
    'ens3_adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'ens3_adv_inception_v3_rename.ckpt'),
    'ens4_adv_inception_v3': os.path.join(FLAGS.checkpoint_path, 'ens4_adv_inception_v3_rename.ckpt'),
    'inception_v4': os.path.join(FLAGS.checkpoint_path, 'inception_v4.ckpt'),
    'inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'inception_resnet_v2_2016_08_30.ckpt'),
    'ens_adv_inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'ens_adv_inception_resnet_v2_rename.ckpt'),
    'resnet_v2_101': os.path.join(FLAGS.checkpoint_path, 'resnet_v2_101.ckpt'),
    'vgg_16': os.path.join(FLAGS.checkpoint_path, 'vgg_16.ckpt'),
    'resnet_v2_152': os.path.join(FLAGS.checkpoint_path, 'resnet_v2_152.ckpt'),
    'adv_inception_resnet_v2': os.path.join(FLAGS.checkpoint_path, 'adv_inception_resnet_v2_rename.ckpt'),
    'resnet_v2_50': os.path.join(FLAGS.checkpoint_path, 'resnet_v2_50.ckpt'),
    'densenet': os.path.join(FLAGS.checkpoint_path, 'tf-densenet161.ckpt'),
    'X101-DA': os.path.join(FLAGS.checkpoint_path, 'X101-DenoiseAll_rename.npz'),
    'R152-B': os.path.join(FLAGS.checkpoint_path, 'R152_rename.npz'),
    'R152-D': os.path.join(FLAGS.checkpoint_path, 'R152_Denoise.npz'),
}

P_kern, kern_size = project_kern(3)
T_kern = gkern(15, 3)


def getLoss(x, y):
    one_hot = tf.one_hot(y, FLAGS.num_classes)

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):  # arg_scope????????????????????????????????????????????????
        logits_v3, end_points_v3 = inception_v3.inception_v3(
            x, num_classes=FLAGS.num_classes, is_training=False, reuse=True)
    auxlogits_v3 = end_points_v3['AuxLogits']

    with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
        logits_v4, end_points_v4 = inception_v4.inception_v4(
            x, num_classes=FLAGS.num_classes, is_training=False, reuse=True)
    auxlogits_v4 = end_points_v4['AuxLogits']

    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits_resnet152, end_points_resnet152 = resnet_v2.resnet_v2_152(
            x, num_classes=FLAGS.num_classes, is_training=False, reuse=True)

    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        logits_Incres, end_points_IR = inception_resnet_v2.inception_resnet_v2(
            x, num_classes=FLAGS.num_classes, is_training=False, reuse=True)
    auxlogits_IR = end_points_IR['AuxLogits']

    flag = 1
    train_model = FLAGS.train_model
    if train_model == 'Inc-v3':
        logits = logits_v3
        auxlogits = auxlogits_v3
    elif train_model == 'Inc-v4':
        logits = logits_v4
        auxlogits = auxlogits_v4
    elif train_model == 'Res152':
        logits = logits_resnet152
        flag = 0
    elif train_model == 'IncRes':
        logits = logits_Incres
        auxlogits = auxlogits_IR

    cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits)
    if flag == 1:
        cross_entropy += tf.losses.softmax_cross_entropy(one_hot, auxlogits)  # TI MI ???PI???
    return cross_entropy


def graph(x, y, i, x_max, x_min, grad, amplification):
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_iter = FLAGS.num_iter
    alpha = eps / num_iter
    momentum = FLAGS.momentum

    # amplification factor
    alpha_beta = alpha * FLAGS.amplification_factor
    gamma = alpha_beta

    # DIM: https://arxiv.org/abs/1803.06978
    # input_diversity(FLAGS, x)
    cross_entropy = getLoss(x, y)

    noise = tf.gradients(cross_entropy, x)[0]

    # TI-FGSM: https://arxiv.org/pdf/1904.02884.pdf
    # noise = tf.nn.depthwise_conv2d(noise, T_kern, strides=[1, 1, 1, 1], padding='SAME')

    # MI-FGSM: https://arxiv.org/pdf/1710.06081.pdf
    # noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
    # noise = momentum * grad + noise

    # Project cut noise
    # amplification += alpha_beta * tf.sign(noise)
    # cut_noise = tf.clip_by_value(abs(amplification) - eps, 0.0, 10000.0) * tf.sign(amplification)
    # projection = gamma * tf.sign(project_noise(cut_noise, P_kern, kern_size))
    #
    # amplification += projection
    # x = x + alpha_beta * tf.sign(noise) + projection

    x = x + alpha * tf.sign(noise)
    x = tf.clip_by_value(x, x_min, x_max)
    i = tf.add(i, 1)

    return x, y, i, x_max, x_min, noise, amplification


def stop(x, y, i, x_max, x_min, grad, amplification):
    num_iter = FLAGS.num_iter
    return tf.less(i, num_iter)


def main(_):
    # Because we normalized the input through "input * 2.0 - 1.0" to [-1,1],
    # the corresponding perturbation also needs to be multiplied by 2
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    num_classes = FLAGS.num_classes

    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        adv_img = tf.placeholder(tf.float32, shape=batch_shape)
        y = tf.placeholder(tf.int32, shape=batch_shape[0])
        x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
        x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

        # v3
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):  # arg_scope????????????????????????????????????????????????
            logits_v3, end_points_v3 = inception_v3.inception_v3(
                adv_img, num_classes=FLAGS.num_classes, is_training=False)
        pre_v3 = tf.argmax(logits_v3, 1)
        # v4
        with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
            logits_v4, end_points_v4 = inception_v4.inception_v4(
                adv_img, num_classes=FLAGS.num_classes, is_training=False)
        pre_v4 = tf.argmax(logits_v4, 1)
        # res152
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits_resnet152, end_points_resnet152 = resnet_v2.resnet_v2_152(
                adv_img, num_classes=FLAGS.num_classes, is_training=False)
        pre_resnet152 = tf.argmax(input=logits_resnet152, axis=1)
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits_resnet101, end_points_resnet101 = resnet_v2.resnet_v2_101(
                adv_img, num_classes=FLAGS.num_classes, is_training=False)
        pre_resnet101 = tf.argmax(input=logits_resnet101, axis=1)
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits_resnet50, end_points_resnet50 = resnet_v2.resnet_v2_50(
                adv_img, num_classes=FLAGS.num_classes, is_training=False)
        pre_resnet50 = tf.argmax(input=logits_resnet50, axis=1)
        # Incres_v2
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits_incres_v2, end_points_incres_v2 = inception_resnet_v2.inception_resnet_v2(
                adv_img, num_classes=FLAGS.num_classes, is_training=False, scope='InceptionResnetV2')
        pre_incres_v2 = tf.argmax(input=logits_incres_v2, axis=1)
        # Inc-v3 adv
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_adv_v3, end_points_adv_v3 = inception_v3.inception_v3(
                adv_img, num_classes=FLAGS.num_classes, is_training=False, scope='AdvInceptionV3')
        pre_adv_v3 = tf.argmax(logits_adv_v3, 1)
        # Inc-v3 ens3
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_ens3_adv_v3, end_points_ens3_adv_v3 = inception_v3.inception_v3(
                adv_img, num_classes=num_classes, is_training=False, scope='Ens3AdvInceptionV3')
        pre_ens3_adv_v3 = tf.argmax(logits_ens3_adv_v3, 1)
        # Inc-v3 ens4
        with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
            logits_ens4_adv_v3, end_points_ens4_adv_v3 = inception_v3.inception_v3(
                adv_img, num_classes=num_classes, is_training=False, scope='Ens4AdvInceptionV3')
        pre_ens4_adv_v3 = tf.argmax(logits_ens4_adv_v3, 1)
        # IncRes_v2 ens
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits_ensadv_incres_v2, end_points_ensadv_incres_v2 = inception_resnet_v2.inception_resnet_v2(
                adv_img, num_classes=FLAGS.num_classes, is_training=False, scope='EnsAdvInceptionResnetV2')
        pre_ensadv_incres_v2 = tf.argmax(input=logits_ensadv_incres_v2, axis=1)

        i = tf.constant(0)
        grad = tf.zeros(shape=batch_shape)
        amplification = tf.zeros(shape=batch_shape)
        x_adv, _, _, _, _, _, _ = tf.while_loop(stop, graph, [x_input, y, i, x_max, x_min, grad, amplification])

        # Run computation ?????????saver??????
        s1 = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))  # ?????????????????????????????????????????????
        s2 = tf.train.Saver(slim.get_model_variables(scope='AdvInceptionV3'))
        s3 = tf.train.Saver(slim.get_model_variables(scope='Ens3AdvInceptionV3'))
        s4 = tf.train.Saver(slim.get_model_variables(scope='Ens4AdvInceptionV3'))
        s5 = tf.train.Saver(slim.get_model_variables(scope='InceptionV4'))
        s6 = tf.train.Saver(slim.get_model_variables(scope='InceptionResnetV2'))
        s7 = tf.train.Saver(slim.get_model_variables(scope='EnsAdvInceptionResnetV2'))  # ???166???scope??????
        s8 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2_101'))  # ???????????????????????????resnet_v2.py??????
        s10 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2_152'))
        s12 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2_50'))

        with tf.Session() as sess:
            s1.restore(sess, model_checkpoint_map['inception_v3'])  # ?????????????????????
            s2.restore(sess, model_checkpoint_map['adv_inception_v3'])
            s3.restore(sess, model_checkpoint_map['ens3_adv_inception_v3'])
            s4.restore(sess, model_checkpoint_map['ens4_adv_inception_v3'])
            s5.restore(sess, model_checkpoint_map['inception_v4'])
            s6.restore(sess, model_checkpoint_map['inception_resnet_v2'])
            s7.restore(sess, model_checkpoint_map['ens_adv_inception_resnet_v2'])
            s8.restore(sess, model_checkpoint_map['resnet_v2_101'])
            s10.restore(sess, model_checkpoint_map['resnet_v2_152'])
            s12.restore(sess, model_checkpoint_map['resnet_v2_50'])
            # s13.restore(sess, model_checkpoint_map['densenet_161'])

            sum_v3, sum_v4, sum_resnet152, sum_resnet101, sum_resnet50, sum_incres_v2, sum_adv_v3, sum_ens3_adv_v3, sum_ens4_adv_v3, sum_ensadv_incres_v2, sum_densenet161 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            adv_l1_diff, adv_l2_diff = 0, 0
            dev = pd.read_csv(FLAGS.input_csv)
            for idx in tqdm(range(0, 1000 // FLAGS.batch_size)):
                images, filenames, True_label = load_images(FLAGS.input_dir, dev, idx * FLAGS.batch_size, batch_shape)
                my_adv_images = sess.run(x_adv, feed_dict={x_input: images, y: True_label}).astype(np.float32)
                pre_v3_, pre_v4_, pre_resnet152_, pre_resnet101_, pre_resnet50_, pre_incres_v2_, pre_adv_v3_, pre_ens3_adv_v3_, pre_ens4_adv_v3_, pre_ensadv_incres_v2_ = \
                    sess.run([pre_v3, pre_v4, pre_resnet152, pre_resnet101, pre_resnet50, pre_incres_v2, pre_adv_v3,
                              pre_ens3_adv_v3, pre_ens4_adv_v3, pre_ensadv_incres_v2],
                             feed_dict={adv_img: my_adv_images})

                sum_v3 += (pre_v3_ != True_label).sum()
                sum_v4 += (pre_v4_ != True_label).sum()
                sum_resnet152 += (pre_resnet152_ != True_label).sum()
                sum_resnet101 += (pre_resnet101_ != True_label).sum()
                sum_resnet50 += (pre_resnet50_ != True_label).sum()
                sum_incres_v2 += (pre_incres_v2_ != True_label).sum()
                sum_adv_v3 += (pre_adv_v3_ != True_label).sum()
                sum_ens3_adv_v3 += (pre_ens3_adv_v3_ != True_label).sum()
                sum_ens4_adv_v3 += (pre_ens4_adv_v3_ != True_label).sum()
                sum_ensadv_incres_v2 += (pre_ensadv_incres_v2_ != True_label).sum()

                # save_images(my_adv_images, filenames, FLAGS.output_dir)
                diff = (my_adv_images + 1) / 2 * 255 - (images + 1) / 2 * 255

                adv_l2_diff += np.mean(np.linalg.norm(np.reshape(diff, [-1, 3]), axis=1))  # np.linalg.norm??????????????????2??????
                adv_l1_diff += np.mean(np.linalg.norm(np.reshape(diff, [-1, 3]), ord=1, axis=1))  # 1??????

    print('{:.1%}'.format(sum_v3 / 1000.0))  # sum_v3
    print('{:.1%}'.format(sum_v4 / 1000.0))  # sum_v4
    print('{:.1%}'.format(sum_resnet152 / 1000.0))  # sum_resnet152
    print('{:.1%}'.format(sum_resnet101 / 1000.0))  # sum_resnet101
    print('{:.1%}'.format(sum_resnet50 / 1000.0))  # sum_resnet50
    print('{:.1%}'.format(sum_incres_v2 / 1000.0))  # sum_incres_v2
    print('{:.1%}'.format(sum_adv_v3 / 1000.0))  # sum_adv_v3
    print('{:.1%}'.format(sum_ens3_adv_v3 / 1000.0))  # sum_ens3_adv_v3
    print('{:.1%}'.format(sum_ens4_adv_v3 / 1000.0))  # sum_ens4_adv_v3
    print('{:.1%}'.format(sum_ensadv_incres_v2 / 1000.0))  # sum_ensadv_Incres_v2
    print('{:.2f}'.format(adv_l1_diff * FLAGS.batch_size / 1000))  # ????????????l1??????
    print('{:.2f}'.format(adv_l2_diff * FLAGS.batch_size / 1000))  # ????????????l2??????


if __name__ == '__main__':
    tf.app.run()
