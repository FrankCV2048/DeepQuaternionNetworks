import numpy as np
from PIL import Image
import os
import random
import operator
import time
import math

BASE_PATH = 'D:\ss'
negative_text = r'negative_pairs_path.txt'
positive_text = r'positive_pairs_path.txt'
triplet_text=r'triplet_aug_pos_neg.txt'


def loadData():
    if ~os.path.exists(negative_text):
        sourcepics = []
        with open(negative_text, 'w') as writer:
            # 遍历源文件
            # 找到所有的源文件目录
            for fpathe, dirs, fs in os.walk('D:\\BaiduYunDownload\\cat'):
                for f in fs:
                    sourcepics.append(os.path.join(fpathe, f))
            # 建立不同对
            k = 0
            for i in range(0, len(sourcepics)):
                for j in range(1,52):
                    s_j=np.random.randint(313)
                    if(i==s_j):
                        s_j=s_j+1
                    if k < 16276:
                        picpairs = sourcepics[i] + '  ' + sourcepics[s_j]
                        writer.writelines(picpairs + '\n')
                        print(k / 276)
                        k = k + 1
                        # 建立文件
    k = 0
    percent = ''
    for k in range(1, 313):
        percent = percent + '||'
    if ~os.path.exists(positive_text):
        # 轮寻原始文件夹，填充不同图片对地址(0-53000)
        with open(positive_text, 'w') as writers:
            for fpathe, dirs, fs in os.walk('D:\\BaiduYunDownload\cat'):
                for f in fs:
                    fname = f.split('.')[0]
                    # 将每一个图片与其对应处理的文件夹中的图片提取出相同图片对
                    for sonfpathe, sondirs, sonfs in os.walk('D:\\BaiduYunDownload\\sss'):
                        for dir in sondirs:
                            if operator.eq(dir, fname):
                                imgpath = os.path.join(sonfpathe, dir)
                                for imgpathe, imgdirs, imgfs in os.walk(imgpath):
                                    imgfs=imgfs[0:33]
                                    for img in imgfs:
                                        picpairs = os.path.join(fpathe, f) + '  ' + os.path.join(imgpathe, img)
                                        writers.writelines(picpairs + '\n')

                                        index = math.ceil(100 * k / 52999)
                                        intpercent = int(10 * index)
                                        print( str(index))
                                        time.sleep(0.001)
                                        k = k + 1
    return 0


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # 形成三元组
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def loadData_triplet_loss():
    k = 0
    percent = ''
    sourcepics = []
    for k in range(1, 1000):
        percent = percent
    if ~os.path.exists(triplet_text):
        for fpathe, dirs, fs in os.walk('X:\cata'):
            for f in fs:
                sourcepics.append(f)
        # 轮寻原始文件夹，填充不同图片对地址(0-53000)
        with open(triplet_text, 'w') as writers:
                for i in range(0,len(sourcepics)):
                    fname = sourcepics[i].split('.')[0]
                    s_j = np.random.randint(1083)
                    if (i == s_j):
                        s_j = s_j + 1
                    negative_path=sourcepics[s_j]
                # 将每一个图片与其对应处理的文件夹中的图片提取出相同图片对
                    for sonfpathe, sondirs, sonfs in os.walk('X:\ss'):
                        for dir in sondirs:
                            if operator.eq(dir, fname):
                                imgpath = os.path.join(sonfpathe, dir)
                                for imgpathe, imgdirs, imgfs in os.walk(imgpath):
                                    for img in imgfs:
                                        picpairs = os.path.join(fpathe, sourcepics[i]) + '  ' + os.path.join(imgpathe, img)+'  '+os.path.join(fpathe,negative_path)
                                        writers.writelines(picpairs + '\n')

                                        index = math.ceil(100 * k / 52999)
                                        intpercent = int(10 * index)
                                        print(percent[:intpercent] + '\n')
                                        print(percent[:intpercent] + '  ' + str(index) + '%' + '\n')
                                        time.sleep(0.001)
                                        k = k + 1

print('sss')

# loadData_triplet_loss()
loadData()
