import os
from PIL import Image
import cv2, math
import numpy as np

# 源目录
myPath = 'E:\\insulator_gross\\yolov5-7.0\\yolov5-7.0\\IDID_XY\\images\\val'
# 输出目录
outPath = 'E:\\insulator_gross\\yolov5-7.0\\yolov5-7.0\\1'


def processImage(filesource, destsource, name, imgtype):
    '''
    filesource是存放待雾化图片的目录
    destsource是存放物化后图片的目录
    name是文件名
    imgtype是文件类型
    '''
    imgtype = 'jpeg' if imgtype == '.jpg' else 'png'
    # 打开图片
    img = cv2.imread(name)
    img_f = img / 255.0
    (row, col, chs) = img.shape

    A = 0.5  # 亮度
    beta = 0.02  # 雾的浓度
    size = math.sqrt(max(row, col))  # 雾化尺寸
    center = (row // 2, col // 2)  # 雾化中心
    for j in range(row):
        for l in range(col):
            d = -0.02 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
            td = math.exp(-beta * d)
            img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
    cv2.imwrite(os.path.join(destsource, name), img_f * 255)


def run():
    # 切换到源目录，遍历目录下所有图片
    os.chdir(myPath)
    for i in os.listdir(os.getcwd()):
        # 检查后缀
        postfix = os.path.splitext(i)[1]
        print(postfix, i)
        if postfix == '.jpg' or postfix == '.png':
            processImage(myPath, outPath, i, postfix)


if __name__ == '__main__':
    run()
