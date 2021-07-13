import cv2
import pytesseract # This is the TesseractOCR Python library
import matplotlib.pyplot as plt
# Set Tesseract CMD path to the location of tesseract.exe file
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

frameWidth = 640
frameHeight = 480
minArea = 200
color = (0, 0, 255)
count = 0
plateCascade = cv2.CascadeClassifier('Resources/haarcascade_russian_plate_number.xml')
text = 'NumberPlate'

cap = cv2.VideoCapture('Resources/vid.mp4')   # 0 means ID of 1st webcam in laptop
cap.set(3, frameWidth)   # 3 is ID of width
cap.set(4, frameHeight)   # 4 is ID of height
cap.set(10, 150)  # 10 is ID of brightness



while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    numberPlates = plateCascade.detectMultiScale(imgGray, 1.1, 10)
    
    for (x, y, w, h) in numberPlates:
        
        area = w*h
        if area > minArea:
            cv2.rectangle(img, (x,y), (x+w, y+h), color, thickness=3)
            cv2.putText(img, text, (x, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
            imgRoi = img[y:y+h, x:x+w]
            cv2.imshow('RoI', imgRoi)
            
            cv2.imwrite('Resources/Scanned/Number_Plate_'+str(count)+'.jpg', imgRoi)
            count += 1
            
    

    img = cv2.resize(img, (1000, 640))
    cv2.imshow('RESULT', img)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        #cv2.rectangle(img, (0,200), (640,300), (0,255,0), cv2.FILLED)
        #cv2.putText(img, 'SAVED', (150,265), cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,255), 2)
        #cv2.imshow('RESULT', img)
        break
        