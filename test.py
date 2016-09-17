__author__ = 'devndraghimire'
__author__ = 'devndraghimire'

import numpy as np
import cv2

from matplotlib import pyplot as plt

img1 = cv2.imread('lambiSmall.png',0)          # queryImage
img2 = cv2.imread('lambicollection.png',0) # trainImage

# Initiate SIFT detector
orb = cv2.ORB_create()

# kp = orb.detect(img1,None)
# kp,descriptors =orb.compute(img1, kp)
# print descriptors

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
#print(matches)
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches[:10], flags=2)

#finalimage= cv2.drawMatchesKnn(def_Image,kp_image,vid_Image,kp_Vid,allmatch,None,flags=2)
plt.imshow(img3),plt.show()
'''
MIN_MATCH_COUNT =10

orb = cv2.ORB()
#video capture
capture = cv2.VideoCapture(0)
#start image processing
def_Image = cv2.imread('you2you.jpg')
changeto_Gray = cv2.cvtColor(def_Image, cv2.COLOR_BGR2GRAY)
kp_image,des_image = orb.detectAndCompute(changeto_Gray,None)
print kp_image

bfmatcher = cv2.BFMatcher()

#For video
while True:
    # Frame retrieving
    ret, vid_Image = capture.read()
    print ret
    gray_Vid = cv2.cvtColor(vid_Image,cv2.COLOR_BGR2GRAY)

    # Keypoint and Descriptors
    kp_Vid, des_Vid = orb.detectAndCompute(gray_Vid,None)
    # kp = orb.detect(gray_Vid,None)
    # print kp
    # matches by bruteforce matcher using knn algorithm

    allmatch = bfmatcher.knnMatch(des_image,des_Vid,k=2)

    # put the matches in Array having the match distance satisfied by 75 %
    good_Match =[]
    for matcha, matchb in allmatch:
        if matcha.distance < 0.75 * matchb.distance:
            good_Match.append(matcha)

    if len(good_Match)>MIN_MATCH_COUNT:
        src_pts = np.float32([kp_image[matcha.queryIdx].pt for matcha in good_Match]).reshape(-1,1,2)
        des_pts = np.float32([kp_Vid[matcha.trainIdx].pt for matcha in good_Match]).reshape(-1,1,2)

        # Create a mask for the homography
        M, mask = cv2.findHomography(src_pts,des_pts,cv2.RANSAC,5.0)
        matched_Mask = mask.ravel().tolist()

        h,w = def_Image.shape[:2]
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        #Draw the perspective Transform box
        dst = cv2.perspectiveTransform(pts,M)

        # Draw lines on the video
        #vid_Image = cv2.polylines(vid_Image,[np.int32(dst)],True,255,3,cv2.LINE_AA)
        print '!!!!!!Matches Found!!!!!!- %d/%d' % (len(good_Match),MIN_MATCH_COUNT)
    #     The freet transform box will show only when the good matches are found

    else:
        print 'Not enough matches Found- %d/%d' % (len(good_Match),MIN_MATCH_COUNT)
        matched_Mask = None

    # draw_params = dict(matchColor = (0,255,0),
    #                singlePointColor = (255,255,0),
    #                matchesMask = matched_Mask,
    #                flags = 2)

    finalimage= cv2.drawMatchesKnn(def_Image,kp_image,vid_Image,kp_Vid,allmatch,None,flags=2)
    # cv2.imshow('win',finalimage)
    cv2.imshow('Windows',vid_Image)

    if cv2.waitKey(10) == 27:
        break
'''
