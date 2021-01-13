1.Develop a Program to display Gray Scale Image Using Read and Write Operation

import cv2 
image=cv2.imread('flower2.jpg') 

cv2.imshow('Original',image) 
cv2.waitKey(0) 

gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
cv2.imwrite('flower1.jpg',gray_image)
cv2.imshow('Grayscale',gray_image)

cv2.waitKey(0) 
cv2.destroyAllWindows() 

OUTPUT:

![image](https://user-images.githubusercontent.com/72332250/104432034-70470280-553d-11eb-9679-0f8a7de8ff44.png)


2. Develop a Program to Perform Linear Transformation on an Image
a) Scaling
import cv2 import numpy as np  
   
FILE_NAME = 'flower2.jpg' try:  
     
    
     
   img = cv2.imread(FILE_NAME)     
   (height, width) = img.shape[:2]     
   cv2.imshow('gulaaaab.jpg', img)  
 
   res = cv2.resize(img, (int(width / 2), int(height / 2)), interpolation = cv2.INTER_CUBIC)      
   # Write image back to disk.     
 cv2.imshow('poooo.jpg', res)      
cv2.waitKey(0)    
except IOError:  
    print ('Error while reading files !!!') 

OUTPUT:

![image](https://user-images.githubusercontent.com/72332250/104432916-5a860d00-553e-11eb-8559-e227f93725a9.png)

b) Rotation


import cv2 import numpy as np  
   
FILE_NAME = 'flower2.jpg' 
try:  
    img = cv2.imread(FILE_NAME)  
   
   (rows, cols) = img.shape[:2]      
   cv2.imshow('gulaaaab.jpg', img)  
   
   M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)      
   res = cv2.warpAffine(img, M, (cols, rows))  
   
   cv2.imshow('result.jpg', res)      
   cv2.waitKey(0)  
 except IOError:  
   print ('Error while reading files !!!')
   
   OUTPUT:
   
![image](https://user-images.githubusercontent.com/72332250/104433461-fa439b00-553e-11eb-839b-4213922b8b3f.png)


3.Develop a Program to find the sum and mean of a set of images 
 Create n number of images and read them from the directoryu and perform the operations
 
 import cv2
import os

path='E:\shradha\imagesip'

imgs = []

files= os.listdir(path)

for file in files:
    
    fpat=path+"\\"+file

    imgs.append(cv2.imread(fpat))
    

i=0

for im in imgs:                                                
#for i in range(len(files)):
    
   cv2.imshow(files[i],imgs[i])
    
   i=i+1;
   cv2.imshow('mean',im/i)
   mean=(im/i)
   print(mean)
   cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
OUTPUT:
![image](https://user-images.githubusercontent.com/72332250/104434897-a3d75c00-5540-11eb-9664-4f89b660d6c9.png)


4. Convert Color Image to gray Scale to Binary Image

import cv2
 image=cv2.imread('flower2.jpg')
 
 cv2.imshow('Original',image)
 cv2.waitKey(0) 
 
gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) cv2.imshow('Grayscale',gray_image) 

cv2.waitKey(0) sqr,binary_image=cv2.threshold(gray_image,172,240,cv2.THRESH_BINARY) cv2.imshow('BinaryImage',binary_image)

cv2.waitKey(0) 
cv2.destroyAllWindows() 

OUTPUT:

![image](https://user-images.githubusercontent.com/72332250/104434090-abe2cc00-553f-11eb-8df3-b5e91b97d503.png)


 



