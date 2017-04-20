import cv2
import numpy as np
from glob import glob
import argparse
from helpers import *
from matplotlib import pyplot as plt


class BOV:
    def __init__(self, no_clusters):
        self.no_clusters = no_clusters
        self.train_path = None
        self.test_path = None
        self.im_helper = ImageHelpers()
        self.bov_helper = BOVHelpers(no_clusters)
        self.file_helper = FileHelpers()
        self.images = None
        self.trainImageCount = 0
        self.train_labels = np.array([])
        self.name_dict = {}
        self.descriptor_list = []
        self.num_keypoints_by_image = np.array([], dtype=int)
        self.saliency_list = []

    def trainModel(self):
        """
        This method contains the entire module 
        required for training the bag of visual words model

        Use of helper functions will be extensive.

        """

        # read file. prepare file lists.
        self.images, self.trainImageCount = self.file_helper.getFiles(self.train_path)
        # extract SIFT Features from each image
        label_count = 0
        self.descriptor_list = None
        for word, imlist in self.images.iteritems():
            word_descriptor_list = None
            word_saliency_list = None
            self.name_dict[str(label_count)] = word
            print label_count + 1, " Computing Features for ", word
            image_count = 0
            for im in imlist:
                # cv2.imshow("im", im)
                # cv2.waitKey()

                self.train_labels = np.append(self.train_labels, label_count)
                kp, des = self.im_helper.features(im)
                sal = self.im_helper.saliency(image=im, keypoints=kp)
                self.num_keypoints_by_image = np.append(self.num_keypoints_by_image, len(kp))

                # Sometimes no keypoints or features can be extracted from the image
                if des is not None:
                    # By using word_descriptor_list the efficiency is improved since
                    # the concatenation of all descriptors is made faster
                    if word_descriptor_list is None:
                        word_descriptor_list = des
                        word_saliency_list = sal
                    else:
                        word_descriptor_list = np.vstack((word_descriptor_list, des))
                        word_saliency_list = np.vstack((word_saliency_list, sal))
                else:
                    print ' - Image ', image_count, ' in category ', word, ' coudln\'t be read.'
                image_count += 1

            if self.descriptor_list is None:
                self.descriptor_list = word_descriptor_list
                self.saliency_list = word_saliency_list
            else:
                self.descriptor_list = np.vstack((self.descriptor_list, word_descriptor_list))
                self.saliency_list = np.vstack((self.saliency_list, word_saliency_list))

            label_count += 1

        # perform clustering
        print "Clustering descriptors"
        #bov_descriptor_stack = self.bov_helper.formatND(self.descriptor_list)
        self.bov_helper.descriptor_vstack = self.descriptor_list
        self.bov_helper.cluster()
        self.bov_helper.developVocabulary(
            n_images=self.trainImageCount,
            saliency_list=self.saliency_list,
            keypoints_by_image=self.num_keypoints_by_image)

        # show vocabulary trained
        # self.bov_helper.plotHist()

        self.bov_helper.standardize()
        self.bov_helper.train(self.train_labels, num_features=100)

    def recognize(self, test_img, test_image_path=None):

        """ 
        This method recognizes a single image 
        It can be utilized individually as well.


        """

        kp, des = self.im_helper.features(test_img)

        # generate vocab for test image
        vocab = np.zeros((1, self.no_clusters))
        # locate nearest clusters for each of 
        # the visual word (feature) present in the image

        # test_ret =<> return of kmeans nearest clusters for N features
        test_ret = self.bov_helper.kmeans_obj.predict(des)

        for each in test_ret:
            vocab[0, each] += 1

        # Scale the features
        vocab = self.bov_helper.scale.transform(vocab)

        # Apply feature selection
        vocab = vocab[:, self.bov_helper.feature_selection]

        # predict the class of the image
        lb = self.bov_helper.clf.predict(vocab)
        # print "Image belongs to class : ", self.name_dict[str(int(lb[0]))]
        return lb

    def testModel(self):
        """ 
        This method is to test the trained classifier

        read all images from testing path 
        use BOVHelpers.predict() function to obtain classes of each image

        """

        self.testImages, self.testImageCount = self.file_helper.getFiles(self.test_path)

        predictions = []
        count2 = 0
        for word, imlist in self.testImages.iteritems():
            print "processing ", word
            for im in imlist:
                cl = self.recognize(im)
                # print self.name_dict[str(int(cl[0]))]
                if self.name_dict[str(int(cl[0]))] == word:
                    count2 += 1
                predictions.append({
                    'image': im,
                    'class': cl,
                    'object_name': self.name_dict[str(int(cl[0]))]
                })
        print self.testImageCount, "Test images"
        print "performance: ", ((count2 * 100) / self.testImageCount)

        # print predictions['object_name']
        # for each in predictions:
        # cv2.imshow(each['object_name'], each['image'])
        # cv2.waitKey()
        # cv2.destroyWindow(each['object_name'])
        #
        # plt.imshow(cv2.cvtColor(each['image'], cv2.COLOR_GRAY2RGB))
        # plt.title(each['object_name'])
        # plt.show()
        # print each['object_name']

    def print_vars(self):
        pass


if __name__ == '__main__':
    # parse cmd args
    parser = argparse.ArgumentParser(
        description=" Bag of visual words example"
    )
    parser.add_argument('--train_path', action="store", dest="train_path", required=True)
    parser.add_argument('--test_path', action="store", dest="test_path", required=True)

    args = vars(parser.parse_args())
    # print args


    bov = BOV(no_clusters=100)

    # set training paths
    bov.train_path = args['train_path']
    # set testing paths
    bov.test_path = args['test_path']
    # train the model
    bov.trainModel()
    # test model
    bov.testModel()
