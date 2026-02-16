# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 10:54:18 2026

@author: bara.fall
"""

#!/usr/bin/python3

import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import sys
#import cv2
#import os

def hk_cluster(image, threshold):
    px_x = np.shape(image)[0]
    px_y = np.shape(image)[1]
    R = np.zeros((px_x+1, px_y+1))
    R[0:px_x, 0:px_y] = np.where(image > threshold, -1, 0)

    last_label = 1
    for j in range(px_y):
        for i in range(px_x):
            if R[i, j] == 0:
                continue
            neighors = np.array([R[i-1, j], R[i, j-1]])
            if any(neighors < 0):
                print("error")
            if np.all(neighors == 0):           # neighors have no labels
                R[i, j] = last_label
                last_label += 1
            elif np.sum(neighors!=0) == 1:      # one neighor has a label
                R[i, j] = np.amax(neighors)
            else:                               # two neighors or more
                R[i, j] = np.amin(neighors, initial=last_label, where=(neighors!=0))
                if R[i, j] == np.amax(neighors, initial=R[i, j], where=(neighors!=0)):
                    continue
                else:
                    R[R == np.amax(neighors, initial=R[i, j], where=(neighors!=0))] = R[i, j]
                    neighors = np.array([R[i-1, j], R[i, j-1]])
                    if R[i, j] == np.amax(neighors, initial=R[i, j], where=(neighors!=0)):
                        continue
                    else:
                        R[R == np.amax(neighors, initial=R[i, j], where=(neighors!=0))] = R[i, j]

    centers = []
    d_mins = []
    d_maxs = []
    diams = []
    diam_stds = []
    orientations = []
    areas = []
#    Image.fromarray(R.astype("uint8"), mode="L").save('test.png')
    for i_count in range(1, last_label):
        if(np.any(R==i_count)):
            y, x = np.nonzero(R==i_count)
            if len(x) < 35:
                continue
            com = [np.mean(x), np.mean(y)]
            d = []
            d_min = px_x + px_y
            d_max = 0
            for x1, y1 in zip(x, y):
                if np.any(R[max(y1-1, 0):min(y1+2, px_x), max(x1-1, 0):min(x1+2, px_y)] != i_count): # attention x and y seem to be interchanged
                    tmp = 2*((x1-com[0])**2+(y1-com[1])**2)**.5
                    if tmp < d_min:
                        d_min = tmp
                    if tmp > d_max:
                        d_max = tmp
                    d.append(tmp)
            if len(d) == 0:
                continue
            diams.append(np.average(d))
            diam_stds.append(np.std(d))
            centers.append(com)
            d_mins.append(d_min)
            d_maxs.append(d_max)
            coords = np.vstack([x, y])
            cov = np.cov(coords)
            evals, evecs = np.linalg.eig(cov)
            sort_indices = np.argsort(evals)[::-1]
            x_v1, y_v1 = evecs[:, sort_indices[1]]  # Eigenvector with smallest eigenvalue
            x_v2, y_v2 = evecs[:, sort_indices[0]]
            evecs[0, 0] = x_v1/(x_v1**2+y_v1**2)**.5
            evecs[0, 1] = y_v1/(x_v1**2+y_v1**2)**.5
            evecs[1, 0] = x_v2/(x_v2**2+y_v2**2)**.5
            evecs[1, 1] = y_v2/(x_v2**2+y_v2**2)**.5
            orientations.append(evecs)
            areas.append(len(x))
    return centers, diams, diam_stds, d_mins, d_maxs, orientations, areas


#def flood_fill(image, threshold):
#    px_x = np.shape(image)[0]
#    px_y = np.shape(image)[1]
#    R = np.zeros((px_x+1, px_y+1))
#    R[0:px_x, 0:px_y] = np.where(image > threshold, -1, 0)

#    px_list = np.argwhere(R == -1)
#    last_label = 1

#    while (len(px_list) != 0):
#        print(last_label)
#        current_particle = np.array([(px_list[last_label, 0], px_list[last_label, 1])])
#        i = 0
#        last_len = 0
#        while(len(current_particle) != i):
#            last_len = len(current_particle)
#            xmin = max(current_particle[i, 0]-2, 0)
#            xmax = min(current_particle[i, 0]+2, px_x+1)
#            ymin = max(current_particle[i, 1]-3, 0)
#            ymax = min(current_particle[i, 1]+3, px_x+1)
#            new = np.argwhere(R[xmin:xmax, ymin:ymax] == -1) + [xmin, ymin]
#            delete = []
#            for j, new_el in enumerate(new):
#                for old_el in current_particle:
#                    if np.array_equal(new_el, old_el):
#                        delete.append(j)
#            new = np.delete(new, delete, axis = 0)
#            current_particle = np.concatenate((current_particle, new))
#            i += 1

#        delete = []
#        for index in current_particle:
#            R[index[0], index[1]] = last_label
#            for j, el in enumerate(px_list):
#                if np.array_equal(index, el):
#                    delete.append(j)
#        px_list = np.delete(px_list, delete, axis = 0)
#        last_label += 1
#        Image.fromarray((np.where(R >= 1, 0, 255)).astype('uint8')).save('a.png')

#    centers = []
#    d_mins = []
#    d_maxs = []
#    diams = []
#    diam_stds = []
#    orientations = []
#    areas = []
#    for i_count in range(1, last_label):
#        if(np.any(R==i_count)):
#            y, x = np.nonzero(R==i_count)
#            if len(x) < 25:
#                continue
#            com = [np.mean(x), np.mean(y)]
#            x = x - np.mean(x)
#            y = y - np.mean(y)
#            d = []
#            d_min = px_x + px_y
#            d_max = 0
#            for x1, y1 in zip(x, y):
#                tmp = 2*np.sqrt(x1**2+y1**2)
#                if tmp < d_min:
#                    d_min = tmp
#                if tmp > d_max:
#                    d_max = tmp
#                d.append(tmp)
#            diams.append(np.average(d))
#            diam_stds.append(np.std(d))
#            centers.append(com)
#            d_mins.append(d_min)
#            d_maxs.append(d_max)
#            coords = np.vstack([x, y])
#            cov = np.cov(coords)
#            evals, evecs = np.linalg.eig(cov)
#            sort_indices = np.argsort(evals)[::-1]
#            x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
#            x_v2, y_v2 = evecs[:, sort_indices[1]]
#            orientations.append([[x_v1/(x_v1**2+y_v1**2)**.5, y_v1/(x_v1**2+y_v1**2)**.5], [x_v2/(x_v2**2+y_v2**2)**.5, y_v2/(x_v2**2+y_v2**2)**.5]])
#    return centers, diams, diam_stds, d_mins, d_maxs, orientations, areas


def adjust_contrast(image):
    image = np.array(image)
    px_x = np.shape(image)[0]
    px_y = np.shape(image)[1]
    block = 150
    for i in range(0, px_x, block):
        for j in range(0, px_y, block):
            xmin = i
            xmax = min(i+block, px_x)
            ymin = j
            ymax = min(j+block, px_y)
            subimage = image[xmin:xmax, ymin:ymax]
            threshold = np.average(subimage)
            subimage = np.where(subimage < threshold-15, 255, 0)
            image[xmin:xmax, ymin:ymax] = subimage
    image = Image.fromarray(image.astype('uint8')).filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MaxFilter(3))
    return image

fct = 3
image = Image.open(r"C:\Users\bara.fall\Downloads\analyse_particlesDan\201805A_B6_017.png")

image2 = image.copy()
image2 = image2.convert('L')
image2b = image2.copy()
image2b = np.array(image2b)
image2 = image2.filter(ImageFilter.BoxBlur(20))
image2 = image2.resize((image2.size[0]//fct, image2.size[1]//fct))
image2 = adjust_contrast(image2)
image3 = image2.filter(ImageFilter.FIND_EDGES)
#image = Image.fromarray(np.maximum(np.array(image3.resize((image.size[0], image.size[1])))[:, :, np.newaxis], np.array(image)))

edges = np.array(image3.resize(image.size))                  # (H, W) en gris
img   = np.array(image.convert("RGB"))                       # (H, W, 3)
edges_rgb = np.repeat(edges[:, :, None], 3, axis=2)          # (H, W, 3)

image = Image.fromarray(np.maximum(edges_rgb, img).astype(np.uint8))


image.save('test.png')
draw = ImageDraw.Draw(image)

centers, diams, diam_stds, d_mins, d_maxs, orientations, areas = hk_cluster(np.array(image2), 100)
i=1
scale = 50/1150
#os.system("rm -rf fft; mkdir fft")
#print("#id diameter_avg_nm diameter_std_nm area_nm2 orientation_x orientation_y")
for center, diam, diam_std, d_min, d_max, orientation, area in zip(centers, diams, diam_stds, d_mins, d_maxs, orientations, areas):
    diam *= fct
    diam_std *= fct
    x = round(fct*center[0])
    y = round(fct*center[1])
    d_min = (diam - diam_std)
    d_max = (diam + diam_std)
#    d_min *= fct
#    d_max *= fct
#    print(i, diam*scale, diam_std*scale,area*fct**2*scale**2, *orientation[0])
    print("xxx", d_min*scale)
    print("xxx", d_max*scale)
    xmin1 = x - 0.5* orientation[0, 0] * d_min
    xmax1 = x + 0.5* orientation[0, 0] * d_min
    ymin1 = y - 0.5* orientation[0, 1] * d_min
    ymax1 = y + 0.5* orientation[0, 1] * d_min
    xmin2 = x - 0.5* orientation[1, 0] * d_max
    xmax2 = x + 0.5* orientation[1, 0] * d_max
    ymin2 = y - 0.5* orientation[1, 1] * d_max
    ymax2 = y + 0.5* orientation[1, 1] * d_max
    draw.line([(xmin1, ymin1), (xmax1, ymax1)], fill="red", width=5)
    draw.line([(xmin2, ymin2), (xmax2, ymax2)], fill="red", width=5)
    draw.ellipse([(x-5, y-5), (x+5, y+5)], outline="blue", width=7)
    draw.text([x, y], str(i)+"\nD (nm): "+str(round(diam*scale, 1))+" +/- "+str(round(diam_std*scale, 1)), fill="black")
#    draw.ellipse([(min(xmin1, xmin2, xmax1, xmax2), min(ymin1, ymin2, ymax1, ymax2)), (max(xmin1, xmin2, xmax1, xmax2), max(ymin1, ymin2, ymax1, ymax2))], outline="blue", width=7)
    xmin = max(round(x - 0.5* d_max), 0)
    xmax = min(round(x + 0.5* d_max), image.size[1])
    ymin = max(round(y - 0.5* d_max), 0)
    ymax = min(round(y + 0.5* d_max), image.size[1])
#    if xmax-xmin > 10 and ymax-ymin > 10:
#        dft = cv2.dft(np.float32(image2b[ymin:ymax, xmin:xmax]), flags = cv2.DFT_COMPLEX_OUTPUT)
#        dft = np.fft.fftshift(dft)
#        dft = 20*np.log(cv2.magnitude(dft[: ,: , 0], dft[:, :, 1]))
#        Image.fromarray(image2b[ymin:ymax, xmin:xmax].astype("uint8"), mode="L").save('fft/direct'+str(i)+'.png')
#        Image.fromarray(dft.astype("uint8"), mode="L").save('fft/fft'+str(i)+'.png')
    i+=1

image.save('result.jpg')

# ======= sauvegarde "labels_preview.png" (particules en couleurs) =======
# On refait un clustering uniquement pour récupérer la matrice R (labels)
px_x = np.shape(np.array(image2))[0]
px_y = np.shape(np.array(image2))[1]
R = np.zeros((px_x+1, px_y+1))
R[0:px_x, 0:px_y] = np.where(np.array(image2) > 100, -1, 0)

last_label = 1
for j in range(px_y):
    for i in range(px_x):
        if R[i, j] == 0:
            continue
        neighors = np.array([R[i-1, j], R[i, j-1]])
        if np.all(neighors == 0):
            R[i, j] = last_label
            last_label += 1
        elif np.sum(neighors != 0) == 1:
            R[i, j] = np.amax(neighors)
        else:
            R[i, j] = np.amin(neighors, initial=last_label, where=(neighors != 0))
            mx = np.amax(neighors, initial=R[i, j], where=(neighors != 0))
            if R[i, j] != mx:
                R[R == mx] = R[i, j]

labels = R[0:px_x, 0:px_y].astype(int)

# Palette couleurs aléatoires (fond blanc)
max_lbl = int(labels.max())
rng = np.random.default_rng(0)
palette = rng.integers(0, 255, size=(max_lbl + 1, 3), dtype=np.uint8)
palette[0] = [255, 255, 255]

# Les pixels "-1" (non encore labelisés) -> fond blanc
labels = np.where(labels < 0, 0, labels)

label_rgb = palette[labels]
label_img = Image.fromarray(label_rgb, mode="RGB")

# même taille que l'image originale
label_img = label_img.resize(image.size, resample=Image.NEAREST)
label_img.save("labels_preview.png")
# ======================================================================
