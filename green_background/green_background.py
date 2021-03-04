"""
Sarit Divekar - 327373684
Hadar Bar-Oz - 204460737
"""
import cv2
import numpy as np

## define images path
input: str = "input.jpg"
background: str = "background.jpg"

## Read
img = cv2.imread(input)
background = cv2.imread(background)

## test reading img is unsuccsessful
if img is None:
  print(f"{input} does not exist in this directory")
## test reading background is unsuccsessful
elif background is None:
  print(f"{background} does not exist in this directory")
else:
    ## convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ## mask of green (30, 70, 70), ~ (90, 255, 255)
    light_green = np.uint8([30, 70, 70])
    dark_green = np.uint8([90, 255, 255])
    mask = cv2.inRange(hsv, light_green, dark_green)

    ## resizing background to size of mask (img)
    background_resized = cv2.resize(background, (mask.shape[1], mask.shape[0]))

    ## crop background to green in img
    background_in_green = cv2.bitwise_and(background_resized, background_resized, mask=mask)

    ## inverse mask for img without the green screen
    mask_inv = cv2.bitwise_not(mask)
    img_not_green = cv2.bitwise_and(img, img, mask=mask_inv)

    ## merging of img with background
    final_img = cv2.bitwise_or(background_in_green, img_not_green)

    ## save
    cv2.imwrite("output.jpg", final_img)

    # cv2.imshow("final", final_img)
    # cv2.waitKey(0)
