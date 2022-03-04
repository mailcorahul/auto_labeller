import os
import cv2

import torch
import torchvision.models as models

class ClassificationLabeler:

    def __init__(
        self,
        class_txt_path,
        reference_dump_path,
        unlabelled_dump_path,
        result_path
        ):

        self.classes = self.create_class_names(class_txt_path)
        self.reference_dump_path = reference_dump_path
        self.class_reference_embeddings = []
        self.model = self.create_backbone(architecture='resnet50')


    def create_class_names(self, class_txt_path):
        """Create a list of class names to label."""

        with open(class_txt_path) as f:
            classes = f.read().split('\n')
            classes = list(filter(None, classes))

        return classes

    def create_backbone(self, architecture='resnet50'):
        """Create a pretrained model object for labelling."""

        model = getattr(models, architecture)(pretrained=True)
        return model


    def create_batch(self, image_paths, batch_size=128):
        """Creates a batch of images with appropriate input transforms."""

        batch = []
        for image_path in image_paths:
            image = cv2.imread(image_path)

            yield batch

    def create_reference_embeddings(self):
        """Create reference embeddings for every class."""

        # load reference images for all classes
        reference_images, class_indices = [], []
        for class_idx, class_label in enumerate(self.classes):
            files = os.listdir(os.path.join(self.reference_dump_path, class_label))
            for file in files:
                image = cv2.imread(os.path.join(self.reference_dump_path, class_label, file))
                reference_images.append(image)
                class_indices.append(class_idx)

        # generate embeddings for all classes




    def label_data(self):
        """Labels image classification data based on the inputs config params."""

        self.create_reference_embeddings()
