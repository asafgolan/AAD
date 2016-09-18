import numpy as np
import cv2

MIN_MATCH_COUNT = 10


from matplotlib import pyplot as plt

# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks = 50)
lambiCounter = 0
#img1 = cv2.imread('just1.jpg',0) # trainImage
#img2 = cv2.imread('real.png',0) # trainImage
img1 = cv2.imread('lambi-logo-2.png',0)          # queryImage
img2 = cv2.imread('lambiCollection.png',0) # trainImage
x = 1
while x == 1:

    cv2.ocl.setUseOpenCL(False)
    # Initiate SIFT detector
    surf = cv2.xfeatures2d.SURF_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = surf.detectAndCompute(img1,None)
    kp2, des2 = surf.detectAndCompute(img2,None)
    #print 'Des1',des2


    # flann = cv2.FlannBasedMatcher(index_params,{ })

    # matches = flann.knnMatch(des1,des2,k=2)
    bfmatcher = cv2.BFMatcher()
    matches = bfmatcher.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        lambiCounter = lambiCounter + 1
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        #img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        img2 = cv2.fillPoly(img2,[np.int32(dst)],255,8, 0)

    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        print 'number of lambi packs are ' , lambiCounter
        matchesMask = None
        x = 0

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)


    #print cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

    #crop_img = img3[92:224, 267:570]

    #plt.imshow(img3, 'gray'),plt.show()
    plt.imshow(img2, 'gray'),plt.show()
