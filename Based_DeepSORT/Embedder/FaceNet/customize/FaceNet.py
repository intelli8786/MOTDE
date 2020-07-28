import tensorflow as tf
import cv2

from ..src import facenet
from ..src.align import detect_face


'''
20170512 : 유클리드거리, 코사인거리 비교
20180402 : SVM 임베딩 비교용
https://github.com/davidsandberg/facenet/issues/948
'''

class FaceNet:
    def __init__(self, model=None):
        with tf.Graph().as_default():
            if model is None:
                print("set up FaceNet weight path!")

            facenet.load_model(model)

            self.ph_images = tf.get_default_graph().get_tensor_by_name("input:0")
            self.ph_phase_train = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            self.model = tf.get_default_graph().get_tensor_by_name("embeddings:0")

            self.sess = tf.Session()


    def __del__(self):
        self.sess.close()

    def faceEmbedding(self, image):
        '''
        https://github.com/davidsandberg/facenet/blob/master/contributed/face.py
        facenet 공식 git
        face.py의 generate_embedding 함수 참조

        '''
        image = cv2.resize(image, (160, 160), interpolation=cv2.INTER_CUBIC)
        image = facenet.prewhiten(image)
        vector = self.sess.run(self.model, feed_dict={self.ph_images: [image], self.ph_phase_train: False})
        return vector[0]