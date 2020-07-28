from .. import model
import tensorflow as tf
import tensorlayer as tl
import cv2
import numpy as np

class SRGAN:
    def __init__(self,  weightPath):
        graph = tf.Graph()
        with graph.as_default():
            self.ph_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')
            self.generator = model.SRGAN_g(self.ph_image, is_train=False, reuse=False)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            tl.layers.initialize_global_variables(self.sess)
            tl.files.load_and_assign_npz(sess=self.sess, name=weightPath, network=self.generator)

    def __del__(self):
        self.sess.close()




    def SuperResolution(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = (image / 127.5) - 1
        out = self.sess.run(self.generator.outputs, {self.ph_image: [image]})[0]
        out = (out + 1) * 127.5
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        out = np.array(out, np.uint8)
        return out
