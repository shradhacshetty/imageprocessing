1.Develop a Program to display Gray Scale Image Using Read and Write Operation

import cv2 
image=cv2.imread('flower2.jpg') cv2.imshow('Original',image) 
cv2.waitKey(0) gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) cv2.imwrite('flower1.jpg',gray_image)
cv2.imshow('Grayscale',gray_image)
cv2.waitKey(0) 
cv2.destroyAllWindows() 

OUTPUT:

![image](https://user-images.githubusercontent.com/72332250/104432034-70470280-553d-11eb-9679-0f8a7de8ff44.png)


2. Develop a Program to Perform Linear Transformation on an Image
a) Scaling



