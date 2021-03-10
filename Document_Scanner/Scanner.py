
# import the necessary packages
import numpy as np
import sys
import cv2
# maintains the aspect ratio
import imutils

def order_points(pts):
	# list of coordinates
	# first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

################################################################################

# construct the argument parser and parse the arguments
cmd_path = sys.argv
imagepath = str(cmd_path[1])
output_path = str(cmd_path[2])
# reading the image from the path
image = cv2.imread(imagepath, cv2.IMREAD_COLOR)
# convert the image to grayscale, blur it, and find edges in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# remove noise - 5 = size of kernel
blurred = cv2.medianBlur(gray, 5)
# Canny Edge Detection
edged = cv2.Canny(blurred, 60, 50, None, 3)
# apply morphological transformations 
dilation = cv2.dilate(edged, np.ones((7,7),np.uint8), iterations = 1)
cv2.imshow("Original", imutils.resize(image, height = 650))
cv2.imshow("dilation", imutils.resize(dilation, height = 650))
cv2.waitKey(0)
cv2.destroyAllWindows()
####################################################################

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
# contours is a Python list of all the contours in the image.
# Each individual contour is a Numpy array of (x,y) coordinates of boundary points of the object.
contours, _ = cv2.findContours(dilation.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key = cv2.contourArea, reverse = True)
# loop over the contours
for c in contours:
	# approximate the contour
	epsilon = 0.1*cv2.arcLength(c,True)
	approx = cv2.approxPolyDP(c,epsilon,True)

	# if our approximated contour has four points, then we
	# can assume that we have found the document in the image.
	if len(approx) == 4:
		screenCnt = approx
		break
# show the contour (outline) of the piece of paper
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Original", imutils.resize(image, height = 650))
cv2.waitKey(0)
cv2.destroyAllWindows()
###########################################################################

# get image height and width
rows, cols = image.shape[:2]
# apply the four point transform to obtain a top-down
points1 = order_points(screenCnt.reshape(4, 2))
# view of the original image
points2 = np.float32([[0, 0], [cols, 0], [cols, rows], [0,rows]])
# get the needed transformation, birds eye view
# H - transformation matrix
H = cv2.getPerspectiveTransform(points1,points2)
# We pass in the image , our transform matrix H , along with the width and height of our output image.
transformed = cv2.warpPerspective(image, H, (cols, rows))

# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect
transformed_gray = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
# apply blur to reduce noise
blurred = cv2.GaussianBlur(transformed_gray, (7, 7), 0)
# create binary image
# The threshold value is a gaussian-weighted sum of the neighbourhood values minus the constant C
adaptive_gaussian = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
# save the final image
cv2.imwrite(output_path, adaptive_gaussian)

# show the original and scanned images
# cv2.imshow("Original", imutils.resize(image, height = 650))
# cv2.imshow("Scanned", imutils.resize(adaptive_gaussian, height = 650))
# cv2.waitKey(0)
