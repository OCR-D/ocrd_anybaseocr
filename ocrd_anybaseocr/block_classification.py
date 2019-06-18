'''
Main Author: Redy Andriyansh
This script is the prepared by Redy Andriyansh as a part of his Master thesis work. The Master thesis details are as follows:

Title of the Thesis: Document Semantic Structure Extraction
Author of the Thesis: Redy Andriyansh
Thesis submission Date: September 2018
Thesis Supervisors:
Dr.-Ing. Syed Saqib Bukhari 
Prof. Dr. Prof. h.c. Andreas Dengel 

Comments Editted by: Syed Saqib Bukhari
Email: saqib.bukhari@dfki.de

Edited for OCRD by: Martin Jenckel

example useage:
python block_segmentation.py --initial_meta_to_learn /path/to/model/*.ckpt.meta --initial_weight_to_learn /path/to/model/*.ckpt --input_path /path/to/img_data --results_csv /path/to/output/result.csv --dataset_mean_pixels x y z
download model from:
https://cloud.dfki.de/owncloud/index.php/s/r9zjrDEwaNczfLN
'''

from __future__ import division
import argparse
import src
from src.finetune import finetune
import glob
import os

parser = argparse.ArgumentParser(description='Optional app description')
# Required positional argument
parser.add_argument('--initial_meta_to_learn', type=str,
                    dest="initial_meta_to_learn",
                    help='A required metafile to restore')

parser.add_argument('--initial_weight_to_learn', type=str,
                    dest="initial_weight_to_learn",
                    help='A required checkpoint to restore')

parser.add_argument('--input_path', type=str,
                    dest="input_path",
                    help='path to the folder with the input images')

parser.add_argument('--results_csv', type=str,
                    dest="results_csv",
                    help='A required file to save results')

parser.add_argument('--batch_size', type=int,
                    dest="batch_size",
                    default=1,
                    help='testing batch size')

parser.add_argument('--num_classes', type=int,
                    dest="num_classes",
                    default=21,
                    help='number of class to classify')

parser.add_argument('--dataset_mean_pixels', type=float,
                    nargs="+",
                    dest="dataset_mean_pixels",
                    help='dataset mean pixel to center the data')


args = parser.parse_args()
finetuner=finetune()

device="gpu"
training_flag=False
file_list = []

for file_path in glob.glob(os.path.join(args.input_path,"*.png")):
    file_list.append(file_path)   

finetuner.test(meta_file=args.initial_meta_to_learn,
               checkpoint_file=args.initial_weight_to_learn,
               file_list=file_list,
               batch_size=args.batch_size,
               num_classes=args.num_classes,
               device=device,
               training_flag=training_flag,
               mean_pixels=args.dataset_mean_pixels,
               results_csv=args.results_csv)
