import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from resizeimage import resizeimage
from PIL import Image

# start timer
start = datetime.now()
print("Start:", start.time())

# construct the argument parse and parse the arguments
# Read the image
cmd_path = sys.argv
left_path = str(cmd_path[1])
right_path = str(cmd_path[2])
output_path = str(cmd_path[3])

##### step 1 #####

# import path of images
left_image = Image.open(left_path)
right_image = Image.open(right_path)

# check if height of left image and right image is the same
min_height = min(left_image.size[1], right_image.size[1])

# resize both images to min height will keeping ratio
left = resizeimage.resize_height(left_image, int(min_height * 0.70))
right = resizeimage.resize_height(right_image, int(min_height * 0.70))

# conver to BGR
left = cv2.cvtColor(np.asarray(left), cv2.COLOR_RGB2BGR)
right = cv2.cvtColor(np.asarray(right), cv2.COLOR_RGB2BGR)

# convert to gray scale
gray_left_image = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
gray_right_image = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

# create an object for extracting SIFT points of interest and their outlines
sift = cv2.SIFT_create()
key_left, descriptors_left = sift.detectAndCompute(gray_left_image, None)
key_right, descriptors_right = sift.detectAndCompute(gray_right_image, None)

##### step 2 #####

# feature matching
matcher = cv2.BFMatcher()
raw_matches = matcher.knnMatch(descriptors_left, descriptors_right, k=2)

##### step 3#####

# apply ratio test
matches = []
ratio = 0.85
for m1, m2 in raw_matches:
    if m1.distance < ratio * m2.distance:
        matches.append(m1)
imMatches = cv2.drawMatches(left, key_left, right, key_right, matches, None)
# plt.imshow(imMatches)
# plt.show()

##### step 4 #####
# Extract location of good matches
left_image_kp = np.float32([key_left[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
right_image_kp = np.float32([key_right[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
# computing a homography requires at least 4 matches
H, status = cv2.findHomography(right_image_kp, left_image_kp, cv2.RANSAC, 5.0)

##### step 5 #####
res = cv2.warpPerspective(right, H, ((right.shape[1] + left.shape[1]), left.shape[0]))

##### step 6 #####
res[0:left.shape[0], 0:left.shape[1]] = left

##### Black background removal #####

# get gray scale image
gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
# threshold for black pixels
_, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
# find non black pixels region
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
# create rectangle of non black pixels
x, y, w, h = cv2.boundingRect(cnt)
# crop rectangle
crop = res[y:y+h, x:x+w]
# write image
cv2.imwrite(output_path, crop)
crop = cv2.cvtColor(np.asarray(crop), cv2.COLOR_BGR2RGB)
plt.imshow(crop)
plt.show()

# end timer
end = datetime.now()
print("End:", end.time())
print(("Total run time:",end-start))
