import tensorflow as tf
import tensorlayer as tl
import cv2
import numpy as np
import os
import re
import random
import scipy.misc
import time

from superResolution.model import SRGAN_g, SRGAN_d, Vgg19_simple_api

batchSize = 12


beta1 = 0.9
epoch = 0
epoch_init = 50
epoch_end = 2000
targetSize = 384
inputSize = 96


lr_init = 1e-4
lr_decay = 0.1
lr_decay_term = 300


path_ckpt = "checkpoint"
path_input = r"./input"
path_target = r"./target"
path_inferenceInput = r"./inference_input"
path_inferencePredict = r"./inference_predict"


# Check Last Checkpoint
for ckpt in os.listdir(path_ckpt):
    result = re.search('\_(\d*)\.',ckpt)
    if result is not None:
        if result[1].isdecimal():
            if epoch < int(result[1]):
                epoch = int(result[1])


def imageLoader(list_pair):
    images_input = []
    images_target = []
    for pair in list_pair:
        image_input = cv2.imread(pair["input"])
        image_target = cv2.imread(pair["target"])

        height, width = image_input.shape[:2]
        pos_x = random.randint(0, width - targetSize - 1)
        pos_y = random.randint(0, height - targetSize - 1)

        image_input = image_input[pos_y:pos_y + targetSize, pos_x:pos_x + targetSize]
        image_target = image_target[pos_y:pos_y + targetSize, pos_x:pos_x + targetSize]



        image_input = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
        image_input = (image_input / 127.5) - 1
        image_input = cv2.resize(image_input, (inputSize, inputSize), interpolation=cv2.INTER_AREA)
        images_input.append(image_input)

        image_target = cv2.cvtColor(image_target, cv2.COLOR_RGB2BGR)
        image_target = (image_target / 127.5) - 1
        images_target.append(image_target)

    return images_input, images_target


# ---------- Dataset Load ----------

list_pair = []
for name_input_class, name_target_class in zip(os.listdir(path_input), os.listdir(path_target)):

    path_input_class = os.path.join(path_input, name_input_class)
    path_target_class = os.path.join(path_target, name_target_class)

    if not os.path.isdir(path_input_class) or not os.path.isdir(path_target_class):
        continue


    for name_input, name_target in zip(os.listdir(path_input_class),os.listdir(path_target_class)):
        name_pair = {}
        name_pair["input"] = os.path.join(path_input_class,name_input)
        name_pair["target"] = os.path.join(path_target_class,name_target)

        list_pair.append(name_pair)

random.shuffle(list_pair)
del list_pair[0:len(list_pair) % batchSize]


# ---------- Sample Image Generate ----------
list_inference = os.listdir(path_inferenceInput)
images_inference = []
for name_inference in list_inference:
    image_inference = cv2.imread(os.path.join(path_inferenceInput, name_inference))
    image_inference = cv2.cvtColor(image_inference, cv2.COLOR_RGB2BGR)
    image_inference = (image_inference / 127.5) - 1
    images_inference.append(image_inference)




# Model Definition
placeholder_input = tf.placeholder('float32', [batchSize, inputSize, inputSize, 3], name='t_image_input_to_SRGAN_generator')
placeholder_target = tf.placeholder('float32', [batchSize, targetSize, targetSize, 3], name='t_target_image')

net_g = SRGAN_g(placeholder_input, is_train=True, reuse=False)
net_d, logits_real = SRGAN_d(placeholder_target, is_train=True, reuse=False)
_, logits_fake = SRGAN_d(net_g.outputs, is_train=True, reuse=True)


# Content loss
image_vgg_target =  tf.image.resize_images(placeholder_target, size=[224, 224], method=0, align_corners=False)
image_vgg_predict = tf.image.resize_images(net_g.outputs, size=[224, 224], method=0, align_corners=False)
net_vgg, vgg_target_emb = Vgg19_simple_api((image_vgg_target + 1) / 2, reuse=False)
_, vgg_predict_emb = Vgg19_simple_api((image_vgg_predict + 1) / 2, reuse=True)
vgg_loss = 2e-6 * tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)
# GAN loss

d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
d_loss = d_loss1 + d_loss2
g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
mse_loss = tl.cost.mean_squared_error(net_g.outputs, placeholder_target, is_mean=True)
# Total loss
g_loss = mse_loss + vgg_loss + g_gan_loss

# GAN model
with tf.variable_scope('learning_rate'):
    lr_v = tf.Variable(lr_init, trainable=False)

g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)

# Pretrain
g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss, var_list=g_vars)

# SRGAN
g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)
d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)



# ---------- RESTORE MODEL ----------
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tl.layers.initialize_global_variables(sess)

# SRGAN Load
if epoch != 0:
    tl.files.load_and_assign_npz(sess=sess, name=os.path.join(path_ckpt,"SRGAN_G_"+str(epoch)+".npz"), network=net_g)
    tl.files.load_and_assign_npz(sess=sess, name=os.path.join(path_ckpt,"SRGAN_D_"+str(epoch)+".npz"), network=net_d)
    print("Load Epoch : ", epoch)

# VGG Load
npz = np.load("vgg19.npy", encoding='latin1').item()
params = []
for val in sorted(npz.items()):
    W = np.asarray(val[1][0])
    b = np.asarray(val[1][1])
    print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
    params.extend([W, b])
tl.files.assign_params(sess, params, net_vgg)


out = sess.run(net_g.outputs, {placeholder_input: images_inference})
for inference_image, inference_name in zip(out,list_inference):
    inference_image = (inference_image + 1) * 127.5
    inference_image = cv2.cvtColor(inference_image, cv2.COLOR_BGR2RGB)
    inference_image = np.array(inference_image, np.uint8)
    cv2.imwrite(os.path.join(path_inferencePredict, inference_name + "_" + str(epoch) + ".png"), inference_image)

# ---------- Training ----------

# init G
sess.run(tf.assign(lr_v, lr_init))

if epoch < epoch_init:
    for epoch_current in range(0, epoch_init):

        random.shuffle(list_pair)

        epoch_time = time.time()
        total_mse_loss, n_iter = 0, 0
        for idx in range(0, len(list_pair), batchSize):
            step_time = time.time()
            list_pair_batch = list_pair[idx: idx + batchSize]
            images_input, images_target = imageLoader(list_pair_batch)
            print(list_pair_batch)

            errM, _ = sess.run([mse_loss, g_optim_init], {placeholder_input: images_input, placeholder_target: images_target})

            total_mse_loss += errM
            n_iter += 1

            print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (epoch, epoch_init , n_iter, time.time() - step_time, errM))

        out = sess.run(net_g.outputs, {placeholder_input: images_inference})
        for inference_image, inference_name in zip(out, list_inference):
            inference_image = (inference_image + 1) * 127.5
            inference_image = cv2.cvtColor(inference_image, cv2.COLOR_BGR2RGB)
            inference_image = np.array(inference_image, np.uint8)
            cv2.imwrite(os.path.join(path_inferencePredict, inference_name + "_" + str(epoch) + ".png"),
                        inference_image)

        # Store Parameter
        if (epoch != 0) and (epoch % 3 == 0):
            tl.files.save_npz(net_g.all_params, name=os.path.join(path_ckpt,'SRGAN_G_'+str(epoch)+'.npz'), sess=sess)
            tl.files.save_npz(net_d.all_params, name=os.path.join(path_ckpt,'SRGAN_D_'+str(epoch)+'.npz'), sess=sess)
        print("[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (epoch, epoch_init, time.time() - epoch_time, total_mse_loss / n_iter))

        epoch += 1


# train SRGAN
for epoch_current in range(0, epoch_end):

    ## update learning rate
    if epoch != 0 and (epoch % lr_decay_term == 0):
        new_lr_decay = lr_decay ** (epoch // lr_decay_term)
        sess.run(tf.assign(lr_v, lr_init * new_lr_decay))

        print(lr_v,lr_init,new_lr_decay)
        print(" ** lr update : %f (for GAN)" % (lr_init * new_lr_decay))

    elif epoch == 0:
        sess.run(tf.assign(lr_v, lr_init))
        print(" ** lr init : %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, lr_decay_term, lr_decay))

    random.shuffle(list_pair)

    epoch_time = time.time()
    total_d_loss, total_g_loss, n_iter = 0, 0, 0
    for idx in range(0, len(list_pair), batchSize):
        step_time = time.time()
        list_pair_batch = list_pair[idx: idx + batchSize]
        images_input, images_target = imageLoader(list_pair_batch)
        print(list_pair_batch)

        ## update D
        errD, _ = sess.run([d_loss, d_optim],  {placeholder_input: images_input, placeholder_target: images_target})
        ## update G
        errG, errM, errV, errA, _ = sess.run([g_loss, mse_loss, vgg_loss, g_gan_loss, g_optim], {placeholder_input: images_input, placeholder_target: images_target})

        total_d_loss += errD
        total_g_loss += errG
        n_iter += 1

        print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f vgg: %.6f adv: %.6f)" %(epoch, epoch_end, n_iter, time.time() - step_time, errD, errG, errM, errV, errA))

    print("[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (epoch, epoch_end, time.time() - epoch_time, total_d_loss / n_iter, total_g_loss / n_iter))

    out = sess.run(net_g.outputs, {placeholder_input: images_inference})
    for inference_image, inference_name in zip(out, list_inference):
        inference_image = (inference_image + 1) * 127.5
        inference_image = cv2.cvtColor(inference_image, cv2.COLOR_BGR2RGB)
        inference_image = np.array(inference_image, np.uint8)
        cv2.imwrite(os.path.join(path_inferencePredict, inference_name + "_" + str(epoch) + ".png"), inference_image)

    # Store Parameter
    if (epoch != 0) and (epoch % 3 == 0):
        tl.files.save_npz(net_g.all_params, name=os.path.join(path_ckpt, 'SRGAN_G_' + str(epoch) + '.npz'), sess=sess)
        tl.files.save_npz(net_d.all_params, name=os.path.join(path_ckpt, 'SRGAN_D_' + str(epoch) + '.npz'), sess=sess)

    epoch += 1

