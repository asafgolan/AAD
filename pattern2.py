__author__ = 'devndraghimire'
import cv2
import numpy as np
from matplotlib import pyplot as plt
MIN_MATCH_COUNT =20
#import imutils

#FLANN PARAMS
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,{})

surf = cv2.xfeatures2d.SURF_create()
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


img1 = cv2.imread('coinsCollection.jpg')
kp1,des1 = surf.detectAndCompute(img1,None)

while True:
    ret,img2 =cap.read()
    # img2 =cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp2,des2 = surf.detectAndCompute(img2,None)
    # print len(kp2)
    print img2.shape[:2]
    #flann Matcher
    matches = flann.knnMatch(des1,des2, k=2)
    good_Matches = []
    for m,n in matches:
        if m.distance <0.75*n.distance:
            good_Matches.append(m)
    #Compare matches if lengths are good and find the location

    if len(good_Matches)>MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_Matches]).reshape(-1,1,2)
        # print src_pts
        dst_pts = np.float32([kp2[m.queryIdx].pt for m in good_Matches]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5,0)
        matches_Mask = mask.ravel().tolist()
        print(img1.shape[:2])
        print(img2.shape)
        h,w = img1.shape[:2]
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        # print pts
        dst =cv2.perspectiveTransform(pts, M)
        print dst
        img2=cv2.polylines(img2,[np.int32(dst)],True,255,3,cv2.LINE_AA)
        print img2
        print ('MATCHES FOUND!!!! - %d/%d' %(len(good_Matches),MIN_MATCH_COUNT))


    else:
        print 'Not enough matches found - %d/%d' % (len(good_Matches),MIN_MATCH_COUNT)
        matches_Mask = None

    # cv2.imshow('camera',img2)
    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matches_Mask,
                       flags =2)

    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,flags=2)
    cv2.imshow('frame',img3)
    plt.show()
