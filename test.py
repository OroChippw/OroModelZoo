import cv2
import numpy as np
img_path = r"D:\Users\User\Desktop\a.jpg"
img = cv2.imread(img_path)
r = cv2.selectROI('input' , img , False)
print("input :" , r)
roi = img[int(r[1]):int(r[1] + r[3]) , int(r[0]) : int(r[0] + r[2])]
rect_ = (int(r[0]) , int(r[1]) , int(r[2]) , int(r[3]))
mask = np.zeros(img.shape[:2] , dtype=np.uint8)

bg_ = np.zeros((1,65) , np.float64)
fg_ = np.zeros((1,65) , np.float64)

cv2.grabCut(img , mask , rect_ , bg_ , fg_ , 11 , mode=cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask == 1) + (mask == 3) , 255 , 0).astype('uint8')

print(mask2.shape)

result_ = cv2.bitwise_and(img , img , mask=mask2)

cv2.imwrite('result.jpg', result_)
cv2.imwrite('roi.jpg', roi)
 
cv2.imshow('roi', roi)
cv2.imshow("result", result_)
cv2.waitKey(0)
cv2.destroyAllWindows()
