import torch
import os
import pickle
import cv2
import numpy as np
from torch.nn import functional as F
import torchvision.transforms as transforms


def get_all_file_paths(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


class SingleQueryModel:
    def __init__(self, model_name, weight_path, query_path):
        self.model_name = model_name
        self.model = None
        self.weight_path = weight_path
        self.query_path = query_path
        self.initialize_model()

    """
    :param test_path: string path to new test image
    """

    @staticmethod
    def euclidean_squared_distance(input1, input2):
        """Computes euclidean squared distance.

        Args:
            input1 (torch.Tensor): 2-D feature matrix.
            input2 (torch.Tensor): 2-D feature matrix.

        Returns:
            torch.Tensor: distance matrix.
        """
        m, n = input1.size(0), input2.size(0)
        mat1 = torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(m, n)
        mat2 = torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat = mat1 + mat2
        distmat.addmm_(input1, input2.t(), beta=1, alpha=-2)
        return distmat

    @staticmethod
    def cosine_distance(input1, input2):
        """Computes cosine distance.

        Args:
            input1 (torch.Tensor): 2-D feature matrix.
            input2 (torch.Tensor): 2-D feature matrix.

        Returns:
            torch.Tensor: distance matrix.
        """
        input1_normed = F.normalize(input1, p=2, dim=1)
        input2_normed = F.normalize(input2, p=2, dim=1)
        distmat = 1 - torch.mm(input1_normed, input2_normed.t())
        return distmat

    @staticmethod
    def simple_cosine(input1, input2):
        return 1 - np.matmul(np.squeeze(input1), np.squeeze(input2).transpose())

    def test_image(self, test_path):
        query = self.load_image(self.query_path)
        gallery = self.load_image(test_path)
        return self.run(query, gallery)

    def test_matrix(self, img_matrix):
        query = self.load_image(self.query_path)
        return self.run(query, np.array(img_matrix))

    def initialize_model(self):
        self.model = cv2.dnn.readNet(self.weight_path)

    def process(self, img):
        blob = cv2.dnn.blobFromImage(img, 0.5, (128, 256), (128, 128, 128), False, False)
        self.model.setInput(blob)
        res = self.model.forward()
        return cv2.normalize(res, None)

    """
    :param path: string path to image
    :return: Tensor[image_channels, image_height, image_width]
    """

    def load_image(self, path):
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def distance_metric(self, input1, input2):
        return self.simple_cosine(input1, input2)

    def run(self, query, gallery):
        query_features = self.process(query)
        gallery_features = self.process(gallery)
        dist = self.distance_metric(query_features, gallery_features)
        return dist


if __name__ == "__main__":

    MODEL = "youtu"
    directory = 'single_query_test/test'
    #directory = 'reid-data/grid/underground_reid/probe'
    TEST_PATHS = get_all_file_paths(directory)
    WEIGHT_PATH = "pretrained/osnet_x0_5_imagenet.pth"
    #QUERY_PATH = 'reid-data/grid/underground_reid/probe/0005_2_25100_229_94_99_249.jpeg'
    QUERY_PATH = "single_query_test/test/nikita4.png"
    dists = []
    for i, test in enumerate(TEST_PATHS):
        print(f"----------------------Image {i + 1}-----------------------")
        print(test)
        engine_runner = SingleQueryModel(MODEL, WEIGHT_PATH, QUERY_PATH)
        dist = engine_runner.test_image(test)
        print(dist)
        dists.append(dist)
    #print(TEST_PATHS)
    #print(dists)
# dists = np.array(dists)

# THRESHOLD_DIST = 0.01  # HIGHLY suspect prechosen hyperparameter, but dunno how else to tell
# indices = np.where(dists < THRESHOLD_DIST)[0]
# print(indices)
# print(dists[indices])
# matching_paths = np.array(TEST_PATHS)[indices]
# print(matching_paths)
# with open('dist.pkl', 'wb') as handle:
#     pickle.dump(dists, handle, protocol=pickle.HIGHEST_PROTOCOL)
