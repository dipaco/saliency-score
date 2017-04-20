import cv2
import numpy as np
from glob import glob
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
from matplotlib import pyplot as plt


class ImageHelpers:
    def __init__(self):
        if cv2.__version__.startswith('2'):
            self.sift_object = cv2.SIFT()
        else:
            self.sift_object = cv2.FeatureDetector_create('SIFT')

    def gray(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def features(self, image):
        keypoints, descriptors = self.sift_object.detectAndCompute(image, None)
        return [keypoints, descriptors]

    def saliency(self, image, keypoints):
        # TODO: Implement this
        return np.random.rand(len(keypoints), 1)


class BOVHelpers:
    def __init__(self, n_clusters=20):
        self.n_clusters = n_clusters
        self.kmeans_obj = KMeans(n_clusters=n_clusters)
        self.kmeans_ret = None
        self.descriptor_vstack = None
        self.mega_histogram = None
        self.saliency_score = np.zeros(n_clusters)
        self.feature_selection = np.array([])
        self.clf = SVC()

    def cluster(self):
        """
        cluster using KMeans algorithm,

        """
        self.kmeans_ret = self.kmeans_obj.fit_predict(self.descriptor_vstack)

    def developVocabulary(self, n_images, saliency_list, keypoints_by_image, kmeans_ret=None):

        """
        Each cluster denotes a particular visual word
        Every image can be represeted as a combination of multiple
        visual words. The best method is to generate a sparse histogram
        that contains the frequency of occurence of each visual word

        Thus the vocabulary comprises of a set of histograms of encompassing
        all descriptions for all images

        Also the saliency score for every cluster is calculated based on teh saliency of
        each keypoint.
        """

        self.mega_histogram = np.array([np.zeros(self.n_clusters) for i in range(n_images)])
        cluster_saliencies = [np.array([]) for i in range(self.n_clusters)]
        old_count = 0
        for i in range(n_images):
            l = keypoints_by_image[i]
            for j in range(l):
                if kmeans_ret is None:
                    idx = self.kmeans_ret[old_count + j]
                else:
                    idx = kmeans_ret[old_count + j]
                self.mega_histogram[i][idx] += 1
                cluster_saliencies[idx] = np.append(cluster_saliencies[idx], saliency_list[old_count + j])
            old_count += l

        # Calculates the saliency score for every cluster
        for cluster_idx in range(len(cluster_saliencies)):
            total_saliency = cluster_saliencies[cluster_idx].sum()
            mean_saliency = total_saliency / float(cluster_saliencies[cluster_idx].size)
            entropy_saliency = entropy(cluster_saliencies[cluster_idx] / total_saliency) / np.log(cluster_saliencies[cluster_idx].size)
            self.saliency_score[cluster_idx] = mean_saliency * entropy_saliency

        print "Vocabulary Histogram Generated"

    def standardize(self, std=None):
        """

        standardize is required to normalize the distribution
        wrt sample size and features. If not normalized, the classifier may become
        biased due to steep variances.

        """
        if std is None:
            self.scale = StandardScaler().fit(self.mega_histogram)
            self.mega_histogram = self.scale.transform(self.mega_histogram)
        else:
            print "STD not none. External STD supplied"
            self.mega_histogram = std.transform(self.mega_histogram)

    def formatND(self, l):
        """
        restructures list into vstack array of shape
        M samples x N features for sklearn

        """
        vStack = np.array(l[0])
        for remaining in l:
            vStack = np.vstack((vStack, remaining))
        self.descriptor_vstack = vStack.copy()
        return vStack

    def train(self, train_labels, num_features=None):
        """
        uses sklearn.svm.SVC classifier (SVM)


        """

        # Peforms feature selection using the saliency score

        sc_idx = np.argsort(1 - self.saliency_score)

        if num_features is not None:
            self.feature_selection = sc_idx[0:num_features]
        else:
            self.feature_selection = np.arange(self.n_clusters, dtype=int)

        print "Training SVM"
        print "Train labels", train_labels
        self.clf.fit(self.mega_histogram[:, self.feature_selection], train_labels)
        print self.clf
        print "Training completed"

    def _feature_selection(self):
        return np.arange(50)

    def predict(self, iplist):
        predictions = self.clf.predict(iplist)
        return predictions

    def plotHist(self, vocabulary=None):
        print "Plotting histogram"
        if vocabulary is None:
            vocabulary = self.mega_histogram

        x_scalar = np.arange(self.n_clusters)
        y_scalar = np.array([abs(np.sum(vocabulary[:, h], dtype=np.int32)) for h in range(self.n_clusters)])

        print y_scalar

        plt.bar(x_scalar, y_scalar)
        plt.xlabel("Visual Word Index")
        plt.ylabel("Frequency")
        plt.title("Complete Vocabulary Generated")
        plt.xticks(x_scalar + 0.4, x_scalar)
        plt.show()


class FileHelpers:
    def __init__(self):
        pass

    def getFiles(self, path):
        """
        - returns  a dictionary of all files
        having key => value as  objectname => image path

        - returns total number of files.

        """
        imlist = {}
        count = 0
        count_1 = 0
        for each in glob(path + "*"):
            word = each.split("/")[-1]
            print " #### Reading image category ", word, " ##### "
            count_1 = 0
            imlist[word] = []
            for imagefile in glob(path + word + "/*[!.db]"):
                # print "Reading file ", imagefile
                im = cv2.imread(imagefile, 0)
                imlist[word].append(im)
                count += 1
                count_1 += 1
            print " ###", count_1, " ", word, "Images ###"

        return [imlist, count]
