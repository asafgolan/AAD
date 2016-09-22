import numpy as np
import cv2

from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 100
lambiCounter = 0

img1 = cv2.imread('book.png',0)

cap = cv2.VideoCapture(0)

cv2.ocl.setUseOpenCL(False)
# Initiate SIFT detector
surf = cv2.xfeatures2d.SURF_create()
kp1, des1 = surf.detectAndCompute(img1,None)



while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp2, des2 = surf.detectAndCompute(gray,None)

    bfmatcher = cv2.BFMatcher()
    matches = bfmatcher.knnMatch(des1,des2,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    #print "GOOD --> ", good

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
        #img2 = cv2.fillPoly(img2,[np.int32(dst)],255,8, 0)
        print 'number of lambi packs are ' , lambiCounter

    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        print 'number of lambi packs are ' , lambiCounter
        matchesMask = None
        x = 0

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)


    #cv2.polylines(frame,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    frame = cv2.drawMatches(img1,kp1,gray,kp2,good,None,**draw_params)

    #crop_img = img3[92:224, 267:570]

    ##plt.imshow(frame, 'gray'),plt.show()

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()

cv2.destroyAllWindows()
