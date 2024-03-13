import cv2 as cv
from win32api import GetSystemMetrics
import numpy as np
from pytesseract import Output
import pytesseract
import imutils
#THIS IS OK: could not be resolved from source Pylance(reportMissingModuleSource)

#Load the image
img = cv.imread("check.jpg") # REPLACE WITH RANDOM IMAGE OF YOUR CHOOSING

#Detect, Crop, and Rotate the Check
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray, (5, 5), 0)
edges = cv.Canny(gray, 50, 150)
contours, _ = cv.findContours(edges.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv.contourArea, reverse=True)[:5]
for c in contours:
    perimeter = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, 0.02 * perimeter, True)
    if len(approx) == 4:
        rect = cv.minAreaRect(approx)
        box = cv.boxPoints(rect)
        box = np.intp(box)
        width = int(rect[1][0])
        height = int(rect[1][1])
        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")
        M = cv.getPerspectiveTransform(src_pts, dst_pts)
        img = cv.warpPerspective(img, M, (width, height))
        break

# load the input image, convert it from BGR to RGB channel ordering,
# and use Tesseract to determine the text orientation
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
results = pytesseract.image_to_osd(rgb, output_type=Output.DICT)
# display the orientation information
print("[INFO] detected orientation: {}".format(
	results["orientation"]))
print("[INFO] rotate by {} degrees to correct".format(
	results["rotate"]))
print("[INFO] detected script: {}".format(results["script"]))

# rotate the image to correct the orientation
img = imutils.rotate_bound(img, angle=results["rotate"])

#Step-by-step method to find the date on the check
#Step 1: Preprocess the image--convert to grayscale, increase contrast, thresholding
def preprocess_image(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
    return thresh
preprocessed_image = preprocess_image(img)
cv.imshow('Before the bounding box', preprocessed_image)
cv.waitKey(0)

#Step 2: Retrieve the bounding box coordinates of a given word
def get_text_bounding_box(image, text):
    data = pytesseract.image_to_data(image, output_type=Output.DICT)
    print(data['text'])
    text_coordinates = []
    for i in range(len(data['text'])):
        if text.lower() in data['text'][i].lower():
            x = data['left'][i]
            y = data['top'][i]
            w = data['width'][i]
            h = data['height'][i]
            text_coordinates.append((x, y, w, h))
    return text_coordinates
date_bbox_list = get_text_bounding_box(img, 'DATE')
if date_bbox_list:
    date_bbox = date_bbox_list[0]
    print("Found 'DATE' at: ", date_bbox)
    cv.rectangle(preprocessed_image, (date_bbox[0], date_bbox[1]),
                 (date_bbox[0] + date_bbox[2], date_bbox[1] + date_bbox[3]),
                 (0, 255, 0), 2)
    cv.imshow('Image with DATE bounding box', preprocessed_image)
    cv.waitKey(0)
else:
    print("Could not find 'DATE' on the check.")

# show the output image
#cv.imshow("before idk", img)
#cv.imshow("after idk", preprocessed_image)
#cv.waitKey(0)

#Resize the image so it fits (in 90%?) of the screen
"""
screen_width = GetSystemMetrics(0) * 0.9
screen_height = GetSystemMetrics(1) * 0.9
height, width = img.shape[:2]
ratio = screen_width / width
new_width = int(width * ratio)
new_height = int(height * ratio)
if new_height > screen_height:
    ratio = screen_height / new_height
    new_width = int(new_width * ratio)
    new_height = int(new_height * ratio)
img = cv.resize(img, (new_width, new_height))
"""

#cv.imshow("Display window", img)
#k = cv.waitKey(0) # Wait for a keystroke in the window

#save the resized image / update the image file
#cv.imwrite("check.jpg", img)