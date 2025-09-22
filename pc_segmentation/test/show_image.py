import cv2

# read image
img_path = '/home/demo/ros2/object_placement/pointcloud_segmentation_ws/tmp/mask_0_piece.jpg'
image = cv2.imread(img_path)
# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
# cv2.imwrite(img_path, image)
print(image.shape)

font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)    # blue
thickness = 2
image = cv2.putText(image, "Press any key to close the window", org, font, fontScale, color, thickness, cv2.LINE_AA)

# show the image, provide window name first
cv2.imshow('image window', image)

# add wait key. window waits until user presses a key
cv2.waitKey(0)

# and finally destroy/close all open windows
cv2.destroyAllWindows()