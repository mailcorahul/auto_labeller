import os
import argparse

from classification_labeler import ClassificationLabeler

parser = argparse.ArgumentParser()
parser.add_argument('--unlabelled-dump', help='path to unlabelled data dump')
parser.add_argument('--class-txt', help='path to txt file containing class name - every line contains one class name')
parser.add_argument('--use-reference', action='store_true')
parser.add_argument('--reference-dump', help='path to reference images for all classes')
parser.add_argument('--result-dump', help='path to save resultant labelled data')

args = parser.parse_args()


if __name__ == '__main__':

    labeler = ClassificationLabeler(
        args.class_txt,
        args.reference_dump,
        args.unlabelled_dump,
        args.result_dump
    )
    labeler.label_images()