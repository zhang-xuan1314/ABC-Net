import cv2
import matplotlib.pyplot as plt

img = cv2.imread('train_data/images/1/0/CHEMBL1025.png',flags=0)
re2, th_img2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(re2)
plt.subplot(121)
plt.imshow(img,cmap='Greys')
plt.subplot(122)
plt.imshow(th_img2,cmap='Greys')
plt.show()