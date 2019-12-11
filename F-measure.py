# -*- coding: UTF-8 -*-
import numpy as np
import math
from PIL import Image
from cv2 import cv2 as cv
import os
from file_read import ListFilesToTxt

DBL_MIN = 2.2250738585072014e-308
DBL_MAX = 1.7976931348623158e+308
RELATIVE_ERROR_FACTOR = 100.0
DBL_EPSILON = 2.2204460492503131e-016
SQRT2 = 1.41421356237
TH_SCORE = 0.85
outfile="binaries.txt"
wildcard = ".txt .jpg .pgm .png"
thick_type = -1
filepath = ''

# ell_labels[i]
## ell_out[i].x1, ell_out[i].y1, ell_out[i].x2, ell_out[i].y2,
## ell_out[i].cx, ell_out[i].cy, ell_out[i].ax, ell_out[i].bx,
## ell_out[i].theta, ell_out[i].ang_start, ell_out[i].ang_end


## 检查文件
def is_txt_file(in_path):
    if not os.path.isfile(in_path):
        return False
    if in_path is not str and not in_path.endswith('.txt'):
        return False
    return True


def is_pgm_file(in_path):
    if not os.path.isfile(in_path):
        return False
    if in_path is not str and not in_path.endswith('.pgm'):
        return False
    return True


def is_jpg_file(in_path):
    if not os.path.isfile(in_path):
        return False
    if in_path is not str and not in_path.endswith('.jpg'):
        return False
    return True


## tools function
def double_equal(a, b):
    if(a == b):
        return True
    abs_diff = abs(a-b)
    aa = abs(a)
    bb = abs(b)
    if aa > bb:
        abs_max = aa
    else:
        abs_max = bb
    if abs_max < DBL_MIN:
        abs_max = DBL_MIN
    if (abs_diff / abs_max) <= (RELATIVE_ERROR_FACTOR * DBL_EPSILON):
        return True
    else:
        return False


def angle_diff(a, b):
    a = a - b
    while a <= -math.pi:
        a = a + math.pi*2
    while a > math.pi:
        a = a - math.pi*2
    if a < 0.0:
        a = -a
    return a


def angle_diff_signed(a, b):
    a = a - b
    while a <= -math.pi:
        a = a + 2*math.pi
    while a > math.pi:
        a = a - math.pi
    return a


def dis(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)


def min_array_pos(v, sz):
    m = DBL_MAX
    if v == None:
        raise Exception("min_array_pos: null input array.")
    if sz <= 0:
        raise Exception("min_array_pos: size must be strictly positive.")
    for i in range(sz):
        if v[i] < m:
            m = v[i]
            pos = i
    return m, pos


def rosin_point(ell_out, x, y):
    d = []
    x1 = ell_out[0]
    y1 = ell_out[1]
    x2 = ell_out[2]
    y2 = ell_out[3]
    cx = ell_out[4]
    cy = ell_out[5]
    semi_b = ell_out[6]
    semi_a = ell_out[7]
    sita = ell_out[8]
    ang_start = ell_out[9]
    ang_end = ell_out[10]
    full = ell_out[11]
    ##
    xtmp = x - cx
    ytmp = y - cy
    ae2 = semi_a**2
    be2 = semi_b**2
    fe2 = ae2 - be2
    xp = xtmp*math.cos(sita) - ytmp*math.sin(-sita)
    yp = xtmp*math.sin(sita) + ytmp*math.cos(-sita)
    xp2 = xp**2
    yp2 = yp**2
    delta = (xp2 + yp2 + fe2) * (xp2 + yp2 + fe2) - 4 * xp2 * fe2
    A = (xp2 + yp2 + fe2 - math.sqrt(delta))/2.0
    ah = math.sqrt(A)
    bh2 = fe2 - A
    term = (A * be2 + ae2 * bh2)
    xx = ah * math.sqrt(ae2 * (be2 + bh2) / term)
    yy = semi_b * math.sqrt(bh2 * (ae2 - A) / term)

    d.append(dis(xp, yp, xx, yy))
    d.append(dis(xp, yp, xx, -yy))
    d.append(dis(xp, yp, -xx, yy))
    d.append(dis(xp, yp, -xx, -yy))

    min_dis, pos = min_array_pos(d, 4)
    if pos == 0:
        pass
    elif pos == 1:
        yy = -yy
    elif pos == 2:
        xx = -xx
    elif pos == 3:
        xx = -xx
        yy = -yy
    xi = xx * math.cos(sita) - yy*math.sin(sita) + cx
    yi = xx * math.sin(sita) + yy*math.cos(sita) + cy
    return xi, yi


def Evaluation(th_score, img, filepath, image_name):
    ## img原图像
    size = img.shape
    sx = size[0] # height
    sy = size[1] # width
    filename = filepath + 'ground_truth/' + image_name + '.txt'
    if not is_txt_file(filename):
        raise Exception("%s 不是一个txt文件" % filename)
    cx = []
    cy = []
    semi_b = []
    semi_a = []
    sita = []
    img_gt = []
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            pass
            x, y, b, a, s = [float(i) for i in lines.split(',')]
            cx.append(x)
            cy.append(y)
            semi_b.append(b)
            semi_a.append(a)
            sita.append(s)
        pass
        cx = np.array(cx)
        cy = np.array(cy)
        semi_b = np.array(semi_b)
        semi_a = np.array(semi_a)
        sita = np.array(sita)
    sz_gt = len(cx)
    for i in range(len(cx)):
        # 创建全0的图
        img_gt_i = np.full((sx,sy),0, dtype=np.uint8)
        cv.ellipse(img_gt_i, (int(cx[i]), int(cy[i])), (int(semi_a[i]), int(semi_b[i])), sita[i]*(180/math.pi), 0, 360, 255, thick_type)
        img_gt.append(img_gt_i)
    cv.imshow('img_gt_i', img_gt[4])
    ## 创建全0背景下的detected
    filename = filepath + 'method_origin/' + image_name + '.txt'
    if not is_txt_file(filename):
        raise Exception("%s 不是一个txt文件" % filename)
    pass
    label = []
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    cx = []
    cy = []
    semi_b = []
    semi_a = []
    sita = []
    ang_start = []
    ang_end = []
    full = []
    img_detect = []
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            pass
            labell, xx1, yy1, xx2, yy2, x, y, ax, bx, theta, ang_s, ang_e, full_ = [
                float(i) for i in lines.split()]
            label.append(labell)
            x1.append(xx1)
            y1.append(yy1)
            x2.append(xx2)
            y2.append(yy2)
            cx.append(x)
            cy.append(y)
            semi_a.append(ax)
            semi_b.append(bx)
            sita.append(theta)
            ang_start.append(ang_s)
            ang_end.append(ang_e)
            full.append(full_)
        label = np.array(label)
        x1 = np.array(x1)
        y1 = np.array(y1)
        x2 = np.array(x2)
        y2 = np.array(y2)
        cx = np.array(cx)
        cy = np.array(cy)
        semi_a = np.array(semi_a)
        semi_b = np.array(semi_b)
        sita = np.array(sita)
        ang_start = np.array(ang_start)
        ang_end = np.array(ang_end)
    for i in range(len(label)):
        img_detct_i = np.full((sx,sy), 0, dtype=np.uint8)
        # img_gt.append(img_gt_i)
        ell_out = []
        ell_out.append(x1[i])
        ell_out.append(y1[i])
        ell_out.append(x2[i])
        ell_out.append(y2[i])
        ell_out.append(cx[i])
        ell_out.append(cy[i])
        ell_out.append(semi_b[i])
        ell_out.append(semi_a[i])
        ell_out.append(sita[i])
        ell_out.append(ang_start[i])
        ell_out.append(ang_end[i])
        ell_out.append(full[i])
        if double_equal(semi_a[i], semi_b[i]):
            img_detct_i = draw_F_E(img_detct_i, ell_out)
        else:
            img_detct_i = draw_F_C(img_detct_i, ell_out)
        img_detect.append(img_detct_i)
    # cv.imshow("img_detct_i", img_detect[5])
    size_test = len(img_detect)
    # sz_test = min(1000, size_test)
    sz_test = size_test
    overlap = np.full((sz_gt,sz_test),0)
    for r in range(sz_gt):
        for c in range(sz_test):
            if TestOverlap(img_gt[r], img_detect[c], th_score, r, c):
                overlap[r, c] = 255
            else:
                overlap[r, c] = 0
    vec_gt = [False for x in range(0,sz_gt)]
    for i in range(sz_test):
        for j in range(sz_gt):
            if vec_gt[j]:
                continue
            bTest = (overlap[j,i] != 0)
            if bTest:
                vec_gt[j] = True
                break
    tp = Count(vec_gt)
    fn = int(sz_gt) - tp
    fp = size_test - tp
    pr = 0
    re = 0
    fmeasure = 0
    if tp == 0:
        if fp == 0:
            pr = 1
            re = 0
            fmeasure = (2 * pr * re) / (pr + re)
        else:
            pr = 0
            re = 0
            fmeasure = 0
    else:
        pr = tp / (tp+fp)
        re = tp / (tp+fn)
        fmeasure = (2*pr*re) / (pr+re)
    return fmeasure, re, pr
    
    
def Evaluation_p(th_score, img, filepath, image_name):
    ## img原图像
    size = img.shape
    sx = size[0] # height
    sy = size[1] # width
    filename = filepath + 'ground_truth/' + image_name + '.txt'
    if not is_txt_file(filename):
        raise Exception("%s 不是一个txt文件" % filename)
    cx = []
    cy = []
    semi_b = []
    semi_a = []
    sita = []
    img_gt = []
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            pass
            x, y, b, a, s = [float(i) for i in lines.split(',')]
            cx.append(x)
            cy.append(y)
            semi_b.append(b)
            semi_a.append(a)
            sita.append(s)
        pass
        cx = np.array(cx)
        cy = np.array(cy)
        semi_b = np.array(semi_b)
        semi_a = np.array(semi_a)
        sita = np.array(sita)
    sz_gt = len(cx)
    for i in range(len(cx)):
        # 创建全0的图
        img_gt_i = np.full((sx,sy),0, dtype=np.uint8)
        cv.ellipse(img_gt_i, (int(cx[i]), int(cy[i])), (int(semi_a[i]), int(semi_b[i])), sita[i]*(180/math.pi), 0, 360, 255, thick_type)
        img_gt.append(img_gt_i)
    # cv.imshow('img_gt_i', img_gt_i)
    ## 创建全0背景下的detected
    filename = filepath + 'method_propossed/' + image_name + '.txt'
    if not is_txt_file(filename):
        raise Exception("%s 不是一个txt文件" % filename)
    pass
    label = []
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    cx = []
    cy = []
    semi_b = []
    semi_a = []
    sita = []
    ang_start = []
    ang_end = []
    full = []
    img_detect = []
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            pass
            labell, xx1, yy1, xx2, yy2, x, y, ax, bx, theta, ang_s, ang_e, full_ = [
                float(i) for i in lines.split()]
            label.append(labell)
            x1.append(xx1)
            y1.append(yy1)
            x2.append(xx2)
            y2.append(yy2)
            cx.append(x)
            cy.append(y)
            semi_a.append(ax)
            semi_b.append(bx)
            sita.append(theta)
            ang_start.append(ang_s)
            ang_end.append(ang_e)
            full.append(full_)
        label = np.array(label)
        x1 = np.array(x1)
        y1 = np.array(y1)
        x2 = np.array(x2)
        y2 = np.array(y2)
        cx = np.array(cx)
        cy = np.array(cy)
        semi_a = np.array(semi_a)
        semi_b = np.array(semi_b)
        sita = np.array(sita)
        ang_start = np.array(ang_start)
        ang_end = np.array(ang_end)
    for i in range(len(label)):
        img_detct_i = np.full((sx,sy), 0, dtype=np.uint8)
        # img_gt.append(img_gt_i)
        ell_out = []
        ell_out.append(x1[i])
        ell_out.append(y1[i])
        ell_out.append(x2[i])
        ell_out.append(y2[i])
        ell_out.append(cx[i])
        ell_out.append(cy[i])
        ell_out.append(semi_b[i])
        ell_out.append(semi_a[i])
        ell_out.append(sita[i])
        ell_out.append(ang_start[i])
        ell_out.append(ang_end[i])
        ell_out.append(full[i])
        if double_equal(semi_a[i], semi_b[i]):
            img_detct_i = draw_F_E(img_detct_i, ell_out)
        else:
            img_detct_i = draw_F_C(img_detct_i, ell_out)
        img_detect.append(img_detct_i)
    # cv.imshow("img_detct_i", img_detect[5])
    size_test = len(img_detect)
    # sz_test = min(1000, size_test)
    sz_test = size_test
    overlap = np.full((sz_gt,sz_test),0)
    for r in range(sz_gt):
        for c in range(sz_test):
            if TestOverlap(img_gt[r], img_detect[c], th_score, r, c):
                overlap[r, c] = 255
            else:
                overlap[r, c] = 0
    vec_gt = [False for x in range(0,sz_gt)]
    for i in range(sz_test):
        for j in range(sz_gt):
            if vec_gt[j]:
                continue
            bTest = (overlap[j,i] != 0)
            if bTest:
                vec_gt[j] = True
                break
    tp = Count(vec_gt)
    fn = int(sz_gt) - tp
    fp = size_test - tp
    pr = 0
    re = 0
    fmeasure = 0
    if tp == 0:
        if fp == 0:
            pr = 1
            re = 0
            fmeasure = (2 * pr * re) / (pr + re)
        else:
            pr = 0
            re = 0
            fmeasure = 0
    else:
        pr = tp / (tp+fp)
        re = tp / (tp+fn)
        fmeasure = (2*pr*re) / (pr+re)
    return fmeasure, re, pr


def Count( v ):
    counter = 0
    for i in range(len(v)):
        if v[i] :
            counter += 1
    return counter



def TestOverlap(gt, test, th, r, c):
    fOR = cv.countNonZero(cv.bitwise_or(gt, test)) # 并集
    fAND = cv.countNonZero(cv.bitwise_and(gt, test)) #交集
    fsim = fAND / fOR
    return fsim >= th


## draw F-ellipse
def draw_F_E(img, ell_out):
    x1 = ell_out[0]
    y1 = ell_out[1]
    x2 = ell_out[2]
    y2 = ell_out[3]
    cx = ell_out[4]
    cy = ell_out[5]
    semi_b = ell_out[6]
    semi_a = ell_out[7]
    sita = ell_out[8]
    ang_start = ell_out[9]
    ang_end = ell_out[10]
    full = ell_out[11]
    ##
    if full == 1:
        cv.ellipse(img, (int(cx), int(cy)), (int(semi_a), int(semi_b)), sita*(180/math.pi), 0, 360, 255, thick_type)
    else:
        x_1 = x1
        y_1 = y1
        x_2 = x2
        y_2 = y2
        if ((double_equal(x_1, x_2) and double_equal(y_1, y_2))
                or dis(x_1, y_1, x_2, y_2) < 2.0):
            cv.ellipse(img, (int(cx), int(cy)), (int(semi_a), int(semi_b)),
                       sita*(180/math.pi), 0, 360, 255, thick_type)
            return img
        ang_start = math.atan2(y_1-cy, x_1-cx) - sita
        ang_end = math.atan2(y_2-cy, x_2-cx) - sita
        if ang_start < 0:
            ang_start += math.pi*2
        if ang_end < 0:
            ang_end += math.pi*2
        if ang_end < ang_start:
            ang_end += math.pi*2
        cv.ellipse(img, (int(cx), int(cy)), (int(semi_a), int(semi_b)), 
                        sita*(180/math.pi), ang_start*(180/math.pi), ang_end*(180/math.pi), 255, thick_type)
        pass
    return img


def draw_F_C(img, ell_out):
    x1 = ell_out[0]
    y1 = ell_out[1]
    x2 = ell_out[2]
    y2 = ell_out[3]
    cx = ell_out[4]
    cy = ell_out[5]
    semi_b = ell_out[6]
    semi_a = ell_out[7]
    sita = ell_out[8]
    ang_start = ell_out[9]
    ang_end = ell_out[10]
    full = ell_out[11]
    # fa = 0
    # fs = 1
    ang_start = math.atan2(y1-cy, x1-cx)
    ang_end = math.atan2(y2-cy, x2-cx)

    C = math.pi * semi_a
    if full or (angle_diff(ang_start, ang_end) < (2*math.pi*SQRT2/C)
                and (angle_diff_signed(ang_start, ang_end)) > 0):
        # 绘制整个圆
        cv.circle(img, (int(cx), int(cy)), int(semi_a), 255, thick_type)
    else:  # 绘制圆弧
        x_1 = semi_a*math.cos(ang_start) + cx
        y_1 = semi_a*math.sin(ang_start) + cy
        x_2 = semi_a*math.cos(ang_end) + cx
        y_2 = semi_a*math.sin(ang_end) + cy
        if ((double_equal(x_1, x_2) and double_equal(y1, y2))
                or (math.sqrt((x_2-x_1)**2 + (y_2-y_1)**2) < 2)):
            cv.ellipse(img, (int(cx), int(cy)), (int(semi_a), int(semi_b)),
                       sita*(180/math.pi), 0, 360, 255, thick_type)
            return img
        ang_start -= sita
        ang_end -= sita
        if ang_start < 0:
            ang_start += 2*math.pi
        if ang_end < 0:
            ang_end += 2*math.pi
        if ang_end < ang_start:
            ang_end += 2*math.pi
        if (ang_end - ang_start) > math.pi:
            fa = 1
        cv.ellipse(img, (int(cx), int(cy)), 
                        (int(semi_a), int(semi_b)),
                        sita*(180/math.pi), 
                        ang_start*(180/math.pi), ang_end*(180/math.pi), 
                        255, thick_type)
    return img


## draw ellipse
def draw_circ_arc(img, ell_out):
    x1 = ell_out[0]
    y1 = ell_out[1]
    x2 = ell_out[2]
    y2 = ell_out[3]
    cx = ell_out[4]
    cy = ell_out[5]
    semi_b = ell_out[6]
    semi_a = ell_out[7]
    sita = ell_out[8]
    ang_start = ell_out[9]
    ang_end = ell_out[10]
    full = ell_out[11]
    # fa = 0
    # fs = 1
    ang_start = math.atan2(y1-cy, x1-cx)
    ang_end = math.atan2(y2-cy, x2-cx)

    C = math.pi * semi_a
    if full or (angle_diff(ang_start, ang_end) < (2*math.pi*SQRT2/C)
                and (angle_diff_signed(ang_start, ang_end)) > 0):
        # 绘制整个圆
        cv.circle(img, (int(cx), int(cy)), int(semi_a), (0, 0, 255))
    else:  # 绘制圆弧
        x_1 = semi_a*math.cos(ang_start) + cx
        y_1 = semi_a*math.sin(ang_start) + cy
        x_2 = semi_a*math.cos(ang_end) + cx
        y_2 = semi_a*math.sin(ang_end) + cy
        if ((double_equal(x_1, x_2) and double_equal(y1, y2))
                or (math.sqrt((x_2-x_1)**2 + (y_2-y_1)**2) < 2)):
            cv.ellipse(img, (int(cx), int(cy)), (int(semi_a), int(semi_b)),
                       sita*(180/math.pi), 0, 360, (0, 0, 255))
            return
        ang_start -= sita
        ang_end -= sita
        if ang_start < 0:
            ang_start += 2*math.pi
        if ang_end < 0:
            ang_end += 2*math.pi
        if ang_end < ang_start:
            ang_end += 2*math.pi
        if (ang_end - ang_start) > math.pi:
            fa = 1
        cv.ellipse(img, (int(cx), int(cy)), 
                        (int(semi_a), int(semi_b)),
                        sita*(180/math.pi), 
                        ang_start*(180/math.pi), ang_end*(180/math.pi), 
                        (0, 0, 255))


def draw_ellipse_arc(img, ell_out):
    x1 = ell_out[0]
    y1 = ell_out[1]
    x2 = ell_out[2]
    y2 = ell_out[3]
    cx = ell_out[4]
    cy = ell_out[5]
    semi_b = ell_out[6]
    semi_a = ell_out[7]
    sita = ell_out[8]
    ang_start = ell_out[9]
    ang_end = ell_out[10]
    full = ell_out[11]
    ##
    if full == 1:
        cv.ellipse(img, (int(cx), int(cy)), (int(semi_a), int(semi_b)), sita*(180/math.pi), 0, 360, (0, 255, 0))
    else:
        x_1 = x1
        y_1 = y1
        x_2 = x2
        y_2 = y2
        if ((double_equal(x_1, x_2) and double_equal(y_1, y_2))
                or dis(x_1, y_1, x_2, y_2) < 3.0):
            cv.ellipse(img, (int(cx), int(cy)), (int(semi_a), int(semi_b)),
                       sita*(180/math.pi), 0, 360, (0, 255, 0))
            return
        ang_start = math.atan2(y_1-cy, x_1-cx) - sita
        ang_end = math.atan2(y_2-cy, x_2-cx) - sita
        if ang_start < 0:
            ang_start += math.pi*2
        if ang_end < 0:
            ang_end += math.pi*2
        if ang_end < ang_start:
            ang_end += math.pi*2
        cv.ellipse(img, (int(cx), int(cy)), (int(semi_a), int(semi_b)), 
                        sita*(180/math.pi), ang_start*(180/math.pi), ang_end*(180/math.pi), (0, 255, 0))
        pass


def show_detected_Ellipses(img, filepath, img_name):
    filename = filepath + 'method_origin/'+img_name
    if not is_txt_file(filename):
        raise Exception("%s 不是一个txt文件" % filename)
    pass
    label = []
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    cx = []
    cy = []
    semi_b = []
    semi_a = []
    sita = []
    ang_start = []
    ang_end = []
    # ell_out = []
    full = []
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            pass
            labell, xx1, yy1, xx2, yy2, x, y, ax, bx, theta, ang_s, ang_e, full_ = [
                float(i) for i in lines.split()]
            label.append(labell)
            x1.append(xx1)
            y1.append(yy1)
            x2.append(xx2)
            y2.append(yy2)
            cx.append(x)
            cy.append(y)
            semi_a.append(ax)
            semi_b.append(bx)
            sita.append(theta)
            ang_start.append(ang_s)
            ang_end.append(ang_e)
            full.append(full_)
        label = np.array(label)
        x1 = np.array(x1)
        y1 = np.array(y1)
        x2 = np.array(x2)
        y2 = np.array(y2)
        cx = np.array(cx)
        cy = np.array(cy)
        semi_a = np.array(semi_a)
        semi_b = np.array(semi_b)
        sita = np.array(sita)
        ang_start = np.array(ang_start)
        ang_end = np.array(ang_end)
        full = np.array(full)
    for i in range(len(label)):
        ell_out = []
        ell_out.append(x1[i])
        ell_out.append(y1[i])
        ell_out.append(x2[i])
        ell_out.append(y2[i])
        ell_out.append(cx[i])
        ell_out.append(cy[i])
        ell_out.append(semi_b[i])
        ell_out.append(semi_a[i])
        ell_out.append(sita[i])
        ell_out.append(ang_start[i])
        ell_out.append(ang_end[i])
        ell_out.append(full[i])
        if double_equal(semi_a[i], semi_b[i]):
            draw_circ_arc(img, ell_out)
        else:
            draw_ellipse_arc(img, ell_out)


def show_detected_Ellipses_propossed(img, filepath, img_name):
    filename = filepath + 'method_propossed/'+img_name
    if not is_txt_file(filename):
        raise Exception("%s 不是一个txt文件" % filename)
    pass
    label = []
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    cx = []
    cy = []
    semi_b = []
    semi_a = []
    sita = []
    ang_start = []
    ang_end = []
    # ell_out = []
    full = []
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            pass
            labell, xx1, yy1, xx2, yy2, x, y, ax, bx, theta, ang_s, ang_e, full_ = [
                float(i) for i in lines.split()]
            label.append(labell)
            x1.append(xx1)
            y1.append(yy1)
            x2.append(xx2)
            y2.append(yy2)
            cx.append(x)
            cy.append(y)
            semi_a.append(ax)
            semi_b.append(bx)
            sita.append(theta)
            ang_start.append(ang_s)
            ang_end.append(ang_e)
            full.append(full_)
        label = np.array(label)
        x1 = np.array(x1)
        y1 = np.array(y1)
        x2 = np.array(x2)
        y2 = np.array(y2)
        cx = np.array(cx)
        cy = np.array(cy)
        semi_a = np.array(semi_a)
        semi_b = np.array(semi_b)
        sita = np.array(sita)
        ang_start = np.array(ang_start)
        ang_end = np.array(ang_end)
        full = np.array(full)
    for i in range(len(label)):
        ell_out = []
        ell_out.append(x1[i])
        ell_out.append(y1[i])
        ell_out.append(x2[i])
        ell_out.append(y2[i])
        ell_out.append(cx[i])
        ell_out.append(cy[i])
        ell_out.append(semi_b[i])
        ell_out.append(semi_a[i])
        ell_out.append(sita[i])
        ell_out.append(ang_start[i])
        ell_out.append(ang_end[i])
        ell_out.append(full[i])
        if double_equal(semi_a[i], semi_b[i]):
            draw_circ_arc(img, ell_out)
        else:
            draw_ellipse_arc(img, ell_out)



def show_ground_truth(img, filepath, image_name):
    filename = filepath + 'ground_truth/' + image_name 
    cx = []
    cy = []
    semi_b = []
    semi_a = []
    sita = []
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            pass
            x, y, b, a, s = [float(i) for i in lines.split(',')]
            cx.append(x)
            cy.append(y)
            semi_b.append(b)
            semi_a.append(a)
            sita.append(s)
        pass
        cx = np.array(cx)
        cy = np.array(cy)
        semi_b = np.array(semi_b)
        semi_a = np.array(semi_a)
        sita = np.array(sita)
    # 在原图上画上gt
    for i in range(len(cx)):
        cv.ellipse(img, (int(cx[i]), int(cy[i])), (int(semi_b[i]), int(semi_a[i])), sita[i]*(180/math.pi), 0, 360, (0, 0, 255),
                   thickness=1, lineType=4)


if __name__ == "__main__":
    # filepath = 'ground_truth\circle1img1.txt'
    # dir = 'images/'
    # file = open(outfile,"w")
    # file_name = ListFilesToTxt(dir,file,wildcard, 1)
    filepath = ''
    ####    数据集
    file_dir = 'images'
    for root, dirs, files in os.walk(file_dir):
        img_name_ = files
    for i in range(len(img_name_)):
        img_name = img_name_[i]
        detected_file = img_name.replace('.pgm','') + '.txt'
        # img_name = 'ring4img3.pgm'
        detected_file = img_name.replace('.pgm','') + '.txt'
        gt_file = img_name.replace('.pgm','') + '.txt'
        # img_file = root + img_name
        img_file = filepath + 'images/' + img_name
        img = cv.imread(img_file)
        img_gt = img.copy()
        img_detected = img.copy()
        img_detected_propossed = img.copy()
        size = img_gt.shape
        # 创建窗口
        cv.namedWindow('img_gt', cv.WINDOW_AUTOSIZE)
        cv.namedWindow('img_detected', cv.WINDOW_AUTOSIZE)
        cv.namedWindow('img_detected_propossed', cv.WINDOW_AUTOSIZE)
        # 加载ground_truth & detected_ellipses
        show_ground_truth(img_gt, filepath, gt_file)
        show_detected_Ellipses(img_detected, filepath, detected_file)
        show_detected_Ellipses_propossed(img_detected_propossed, filepath, detected_file)
        ## 计算指标
        f_score_o, re_o, pr_o = Evaluation( TH_SCORE, img_gt, filepath, img_name.replace('.pgm','') )
        f_score_p, re_p, pr_p = Evaluation_p( TH_SCORE, img_gt, filepath, img_name.replace('.pgm','') )
        f_score_out = ('f_measure_origin/'+img_name.replace('.pgm','')+'.txt')
        # with open('f_measure_origin/f_measure.txt','w') as f:
        with open(f_score_out,'w') as f:
            f.write('img_name = {0}\n'.format(img_name))
            f.write('f_score_o = {0}\tre_o = {1}\tpr_o = {2}\n'.format(f_score_o, re_o,pr_o))
            f.write('f_score_p = {0}\tre_p = {1}\tpr_p = {2}\n'.format(f_score_p, re_p,pr_p))
    print('f_score_o = {0}\tre_o = {1}\tpr_o = {2}\n'.format(f_score_o, re_o, pr_o))
    print('f_score_p = {0}\tre_p = {1}\tpr_p = {2}\n'.format(f_score_p, re_p, pr_p))
    ####    单张图片
    """
    img_name = 'ring4img3.pgm'
    detected_file = img_name.replace('.pgm','') + '.txt'
    gt_file = img_name.replace('.pgm','') + '.txt'
    # img_file = root + img_name
    
    img_file = filepath + 'images/' + img_name
    img = cv.imread(img_file)
    img_gt = img.copy()
    img_detected = img.copy()
    img_detected_propossed = img.copy()
    size = img_gt.shape
    # 创建窗口
    cv.namedWindow('img_gt', cv.WINDOW_AUTOSIZE)
    cv.namedWindow('img_detected', cv.WINDOW_AUTOSIZE)
    cv.namedWindow('img_detected_propossed', cv.WINDOW_AUTOSIZE)
    # 加载ground_truth & detected_ellipses
    show_ground_truth(img_gt, filepath, gt_file)
    show_detected_Ellipses(img_detected, filepath, detected_file)
    show_detected_Ellipses_propossed(img_detected_propossed, filepath, detected_file)
    ## 计算指标
    f_score_o, re_o, pr_o = Evaluation( TH_SCORE, img_gt, filepath, img_name.replace('.pgm','') )
    f_score_p, re_p, pr_p = Evaluation_p( TH_SCORE, img_gt, filepath, img_name.replace('.pgm','') )
    print('f_score_o = {0}\tre_o = {1}\tpr_o = {2}\n'.format(f_score_o, re_o, pr_o))
    print('f_score_p = {0}\tre_p = {1}\tpr_p = {2}\n'.format(f_score_p, re_p, pr_p))
    with open('f_measure_origin/f_measure.txt','w') as f:
        f.write('img_name = {0}\n'.format(img_name))
        f.write('f_score_o = {0}\tre_o = {1}\tpr_o = {2}\n'.format(f_score_o, re_o,pr_o))
        f.write('f_score_p = {0}\tre_p = {1}\tpr_p = {2}\n'.format(f_score_p, re_p,pr_p))
    """
    # show
    cv.imwrite("img_gt.jpg", img_gt)
    cv.imwrite("img_detected.jpg", img_detected)
    cv.imwrite("img_detected_propossed.jpg", img_detected_propossed)
    cv.imshow('img_gt', img_gt)
    cv.imshow('img_detected', img_detected)
    cv.imshow("img_detected_propossed", img_detected_propossed)
    cv.waitKey(0)
