from funcOD import *
from PIL import Image
import sys

try:

    resultData = mainPreLoadModel('./testfolder/')
    #mainPredict('','./testfolder/', '' ,xx['result'], '', '')
    im = Image.open('./testfolder/test1_1.png')
    return mainPredict(im, '/content/yolov7/testfolder/' , '', resultData['result'], '', '')
    
    #mainPredict(im, './testfoler/' , '', resultData['result'], '', '')
    #mainPredict(im, './testfoler/' , '', resultData['result'], '', '')
    #mainPredict(im, './testfoler/' , '', resultData['result'], '', '')
    #mainPredict(im, './testfoler/' , '', resultData['result'], '', '')
    #mainPredict(im, './testfoler/' , '', resultData['result'], '', '')

    #res = test_detector.test(im, './testfolder/','','')
    #print(res)

except:
    print("Unexpected error:", sys.exc_info()[0])

