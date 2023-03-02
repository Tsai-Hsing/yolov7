from ctypes import *
import math
import random
import os
import json
import cv2 
from PIL import Image
import argparse
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

from .models.experimental import attempt_load
from .utils.datasets import create_dataloader
from .utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from .utils.metrics import ap_per_class, ConfusionMatrix
from .utils.plots import plot_images, output_to_target, plot_study_txt
from .utils.torch_utils import select_device, time_synchronized, TracedModel


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(lib, net, meta, niou, nc, data, imgsz, image, thresh=.5, hier_thresh=.5, nms=.45):
    model = net
    training = False
    v5_metric = False
    augment = False
    save_hybrid = False
    compute_loss = None
    conf_thres=0.001
    iou_thres=0.6
    save_txt=False
    save_json=False
    plots = True
    verbose=False
    device = meta
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--task', default='test', help='train, val, test, speed or study')
    #parser.add_argument('--data', type=str, default='testfolder/diamond.yaml', help='*.data path')
    opt = parser.parse_args()
    #imgsz = 1920
    imgsz = check_img_size(imgsz, s=lib)
    batch_size = 1
    # Logging
    log_imgs = 0
    wandb_logger = None
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        task = 'test'  # path to train/val/test images
        dataloader = create_dataloader(data[task], imgsz, batch_size, lib, opt, pad=0.5, rect=True, workers=0,
                                       prefix=colorstr(f'{task}: '))[0]
        print('fsafsd')
    if v5_metric:
        print("Testing with YOLOv5 AP metric...")

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        print(width)
        print(height)
        with torch.no_grad():
            # Run model
            t = time_synchronized()
            out, train_out = model(img, augment=augment)  # inference and training outputs
            t0 += time_synchronized() - t

            # Compute loss
            if compute_loss:
                loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls
            
            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t = time_synchronized()
            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
            t1 += time_synchronized() - t
        
        alltype = []
        res = []
        returnval = output_to_target(out)
        for obj in returnval:
            if obj[6] >= 0.25:
                res.append((data['names'][int(obj[1])], obj[6], (math.floor((obj[2] * 2 - obj[4]) / 2) - shapes[0][1][1][0] , math.floor((obj[3] * 2 - obj[5]) / 2) - shapes[0][1][1][1] , math.ceil((obj[2] * 2 - obj[4]) / 2 + obj[4]) - shapes[0][1][1][0], math.ceil((obj[3] * 2 - obj[5]) / 2 + obj[5]) - shapes[0][1][1][1])))
                if len(alltype) == 0 :
                    alltype.append(data['names'][int(obj[1])])
                if data['names'][int(obj[1])] not in alltype:
                    alltype.append(data['names'][int(obj[1])])
        return res, alltype

def mainPredict(image, path, modelName, userDict , graph, sess):
    data = {}
    data['isSuccess'] = 'true'
    data['ErrorMsg'] = ''
    data['result'] = []
    try:
        print('1')
        path = bytes(path, 'ascii')
        net = userDict['net']
        meta = userDict['meta']
        lib = userDict['lib']
        niou = userDict['niou']
        nc = userDict['nc']
        mdata = userDict['data']
        #imagepath = path.decode() + '/testfolder/original.' + userDict["FileExtension"]#image.format.lower()
        imagepath = os.path.join(path.decode(), "testfolder", "original." + userDict["FileExtension"])
        #imagepath = os.path.abspath(os.getcwd()) + '/original.' + image.format.lower()
        #print(imagepath)
        #os.system('rm -rf ' + path.decode() + '/testfolder/test.cache')
        os.system('rm -rf ' + os.path.join(path.decode(), 'testfolder', 'test.cache'))
        image.save(imagepath)
        print(imagepath)
        imgsz = cv2.imread(imagepath).shape[1]
        r, alltype = detect(lib, net, meta, niou, nc, mdata, imgsz, bytes(imagepath, 'ascii'))   
        infod = {}
        at = []
        for cls, score, bbox in r:
            obj = {
                        'class':cls,
                        'classResult':[{'score': str(score),'BoundingBox':[str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3])],'Value':[str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3])]}]
                        }
            if cls not in at:
                at.append(cls)
                infod[cls] = []
                infod[cls].append(obj)
            else:
                for iobj in infod[cls]:
                    if iobj['class'] == cls:
                        iobj['classResult'].append({'score': str(score),'BoundingBox':[str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3])],'Value':[str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3])]})
        for obj in alltype:
            data['result'].append(infod[obj])
        #print(r)
        print(json.dumps(data))
    except Exception as e:
        print(e)
        error_class = e.__class__.__name__  # 取得錯誤類型
        detail = e.args[0]  # 取得詳細內容
        cl, exc, tb = sys.exc_info()  # 取得Call Stack
        lastCallStack = traceback.extract_tb(tb)[-1]  # 取得Call Stack的最後一筆資料
        fileName = lastCallStack[0] #取得發生的檔案名稱
        lineNum = lastCallStack[1]  # 取得發生的行號
        funcName = lastCallStack[2]  # 取得發生的函數名稱
        errorMsg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
        data['isSuccess']= 'false'
        data['ErrorMsg']= str(errorMsg)
    finally:
        return json.dumps(data)

def mainPreLoadModel(path):
    resultData = {}
    resultData['isSuccess'] = 'true'
    resultData['result'] = {}
    resultData['ErrorMsg'] = ""
    modelDict = {}
    try:
        path = bytes(path, 'ascii')
        modelname = ''
        for filename in os.listdir(os.path.join(path, b'testfolder')):
            if filename.lower().endswith(b'.pt'):
                modelname = os.path.splitext(filename)[0]
        #fin = open(path.decode() + '/testfolder/yolov7.yaml', 'rt')
        fin = open(os.path.join(path.decode(), 'testfolder', 'yolov7.yaml'), 'rt')
        data = fin.read()
        #data = data = data.replace('./testfolder', path.decode() + "/testfolder")
        data = data = data.replace('./testfolder', os.path.join(path.decode(), "testfolder"))
        fin.close()
        #fin = open(path.decode() + '/testfolder/yolov7.yaml', 'wt')
        fin = open(os.path.join(path.decode(), 'testfolder', 'yolov7.yaml'), 'wt')
        fin.write(data)
        fin.close()
        model=None
        # Initialize/load model and set device
        training = model is not None
        if training:  # called by train.py
            device = next(model.parameters()).device  # get model device

        else:  # called directly
            set_logging()
            device = select_device("cpu", batch_size=1)

        # Directories
        #model = attempt_load(path + b"/" + modelname + b".pt", map_location=device)  # load FP32 model
        #model = attempt_load(path.decode() + "/testfolder/" + modelname.decode() +  ".pt", map_location=device)
        model = attempt_load(os.path.join(path.decode(), "testfolder", modelname.decode() +  ".pt"), map_location=device)
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = 1920
        imgsz = check_img_size(imgsz, s=gs)  # check img_size
        trace = False
        if trace:
            model = TracedModel(model, device, imgsz)

        # Half
        half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
        if half:
            model.half()

        # Configure
        model.eval()
        #data = path.decode() + '/testfolder/yolov7.yaml'
        data = os.path.join(path.decode(), 'testfolder', 'yolov7.yaml')
        if isinstance(data, str):
            is_coco = data.endswith('coco.yaml')
            with open(data) as f:
                data = yaml.load(f, Loader=yaml.SafeLoader)
        #check_dataset(data)  # check
        single_cls = False
        nc = 1 if single_cls else int(data['nc'])  # number of classes
        iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()
        
        modelDict['net'] = model
        modelDict['FileExtension'] = "png"
        modelDict['meta'] = device
        modelDict['lib'] = gs
        modelDict['niou'] = niou
        modelDict['nc'] = nc
        modelDict['data'] = data
        resultData['result'] = modelDict
        #print(resultData)
    except Exception as e:
        print(e)
        error_class = e.__class__.__name__  # 取得錯誤類型
        detail = e.args[0]  # 取得詳細內容
        cl, exc, tb = sys.exc_info()  # 取得Call Stack
        lastCallStack = traceback.extract_tb(tb)[-1]  # 取得Call Stack的最後一筆資料
        fileName = lastCallStack[0]  # 取得發生的檔案名稱
        lineNum = lastCallStack[1]  # 取得發生的行號
        funcName = lastCallStack[2]  # 取得發生的函數名稱
        errorMsg = "File \"{}\", line {}, in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
        resultData['isSuccess'] = 'false'
        resultData['ErrorMsg'] = str(errorMsg)
    return resultData
