import argparse
import cv2
import matplotlib.pyplot as plt 


ap = argparse.ArgumentParser()
# ap.add_argument("")

path_image ="/home/asma/Documents/Programing/Essentials/Essentials-/OpenCV/image1.jpeg"
image = cv2.imread(path_image)

plt.imshow(image)

plt.show()