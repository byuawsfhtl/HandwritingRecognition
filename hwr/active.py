import tensorflow as tf
import numpy as np
from tqdm import tqdm

from hwr.util import prediction_confidence, bp_decode
import hwr.dataset as ds


class InstanceSelector:
    """
    Object for actively selecting instances from a dataset for training
    """
    def __init__(self, model, img_path, img_size, sample_size=-1, inference_batch_size=128):
        """

        @param model: The model used during the active learning selection process
        @param img_path: The path to the images that are available for selection
        @param img_size: The size the images will be resized to given the model
        @param sample_size: The number of instances in observable pool for selection. If -1, the entire pool is used
        @param inference_batch_size: The batch size used when performing model inference which is used during the
        selection process
        """
        self.model = model
        self.file_list = np.array(list(ds.get_file_list(img_path).as_numpy_iterator()))
        self.img_size = img_size
        self.sample_size = sample_size
        self.batch_size = inference_batch_size
        self.possible_metrics = ['confidence', 'random']

    def update_model(self, model):
        """
        Update the InstanceSelector with a newer model
        @param model: The model to be used for inference during the selection process
        """
        self.model = model

    @staticmethod
    def create_list_dataset(the_list):
        """
        Static method to create a tf.data.Dataset from a list

        @param the_list: The list to be converted to a tf.data.Dataset
        @return: The tf.data.Dataset
        """
        return tf.data.Dataset.from_tensor_slices(the_list)

    @staticmethod
    def remove_elements(the_list, elems):
        """
        Static method to remove elements from list

        @param the_list: The list containing elements to remove
        @param elems: The indices of the elements to remove
        @return: The list
        """
        return np.delete(the_list, elems, None)

    @staticmethod
    def argshuffle(the_list):
        """
        Static method to shuffle the indices in a list
        @param the_list: The list that requires shuffling
        @return: Indices in the list that have been shuffled
        """
        idxs = list(range(len(the_list)))
        np.random.shuffle(idxs)
        return idxs

    @staticmethod
    def argsort(the_list):
        """
        Static method to sort the indices in a list
        @param the_list:  The list that requires sorting
        @return: Indices in the list that have been sorted
        """
        idxs = np.argsort(the_list)
        return idxs

    def select(self, num_instances, metric='confidence'):
        """
        Actively select a number of instances from the dataset for training.

        @param num_instances: The number of instances to select from the dataset
        @param metric: Which metric to use when selecting instances ['confidence', 'random']
        @return: A tf.data.Dataset containing the selected instances as given from
        hwr.dataset.get_encoded_inference_dataset_...
        """
        if metric == 'confidence':
            dataset = ds.get_encoded_inference_dataset_from_file_list(
                self.create_list_dataset(self.file_list), self.img_size)

            if self.sample_size != -1:
                dataset = dataset.take(self.sample_size)
                dataset_size = int(np.ceil(self.sample_size/self.batch_size))
            else:
                dataset_size = int(np.ceil(len(self.file_list)/self.batch_size))

            dataset = dataset.batch(self.batch_size)

            confidences = []
            prediction_loop = tqdm(total=dataset_size, position=0, leave=True)
            prediction_loop.set_description('Training Instances Selection Progress')
            for index, (img, img_name) in enumerate(dataset):
                output = self.model(img)
                prediction = bp_decode(output)
                confidence = prediction_confidence(output, prediction)
                confidences.extend(confidence)
                prediction_loop.update(1)
            prediction_loop.close()

            idxs = self.argsort(confidences)
        elif metric == 'random':
            idxs = self.argshuffle(self.file_list)
        else:
            raise NotImplementedError('The metric {} is not available. Possible metrics: {}'.format(
                metric, self.possible_metrics))

        selected_file_list = np.take(self.file_list, idxs[:num_instances])
        self.file_list = self.remove_elements(self.file_list, idxs[:num_instances])

        file_list_dataset = self.create_list_dataset(selected_file_list)

        return ds.get_encoded_inference_dataset_from_file_list(file_list_dataset, self.img_size)
