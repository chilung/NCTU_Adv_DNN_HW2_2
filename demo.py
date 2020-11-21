# -*- coding: utf-8 -*-
'''
@Time          : 20/04/25 15:49
@Author        : huguanghao
@File          : demo.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
'''

import os
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import argparse
import cv2
import time
import datetime
import json
import numpy as np

"""hyper parameters"""
use_cuda = True

def output_transform(img_width, img_height, rel_center_x, rel_center_y, rel_width, rel_height):
    center_x = rel_center_x * img_width
    center_y = rel_center_y * img_height
    width = rel_width * img_width
    height = rel_height * img_height
    return (int(round(center_y-height/2)), int(round(center_x-width/2)), int(round(center_y+height/2)), int(round(center_x+width/2)))

def detect_cv2(cfgfile, weightfile, imgfiles, namesfile):
    CENTER_X = 0
    CENTER_Y = 1
    WIDTH = 2
    HEIGHT = 3
    CONFIDENCE = 4
    CLASS_ID = 6
    
    m = Darknet(cfgfile)

    # m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    class_names = load_class_names(namesfile)

    f = open(imgfiles, 'r')
    filelist = f.read().splitlines()
    f.close()
    # print(filelist)
    
    print(time.ctime())
    start_time = time.time()

    submit_results = []
    for idx, imgfile in enumerate(filelist):
        # print(imgfile)
        img = cv2.imread(imgfile)
        sized = cv2.resize(img, (m.width, m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        img_height, img_width, _ = img.shape

        predict = do_detect(m, sized, 0.2, 0.6, use_cuda)[0]
        if idx%100 == 0:
            print('predicting img progress: {} of {}'.format(idx, len(filelist)))

        result = {}
        result['bbox'] = [
            output_transform(
                img_width, img_height,
                box[CENTER_X], box[CENTER_Y],
                box[WIDTH], box[HEIGHT])
            for box in predict]
        # print(result['bbox'])
        result['score'] = [float(box[CONFIDENCE]) for box in predict]
        # print(result['score'])
        result['label'] = [int(box[CLASS_ID]) if not box[CLASS_ID]==0 else 10 for box in predict]
        # print(result['label'])
        # print(result)
        submit_results.append(result)

    end_time = time.time()
    print(time.ctime())
    print(end_time, start_time)
    print('number of predicts: {}, fps: {}'.format(len(submit_results), len(submit_results)/(end_time-start_time)))

    print('output results for submission: {}'.format('submission.json'))
    with open('submission.json', 'w') as outfile:
       json.dump(submit_results, outfile)

    # plot_boxes_cv2(img, boxes[0], savename='predictions.jpg', class_names=class_names)

def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str, default='./exec_env/yolo-obj.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default='./exec_env/yolo-obj-mAP_93.3.weights',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument('-imgfiles', type=str,
                        default='./exec_env/valid.txt',
                        help='path of your image file directory.', dest='imgfiles')
    parser.add_argument('-namesfile', type=str,
                        default='./exec_env/obj.names',
                        help='path of your name file.', dest='namesfile')
    args = parser.parse_args(args=['-cfgfile', './exec_env/yolo-obj.cfg',
                    '-weightfile', './exec_env/yolo-obj-mAP_93.3.weights',
                    '-imgfiles', './exec_env/valid.txt',
                    '-namesfile', './exec_env/obj.names'])
    print(args)
    # args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()
    if args.imgfiles:
        detect_cv2(args.cfgfile, args.weightfile, args.imgfiles, args.namesfile)
