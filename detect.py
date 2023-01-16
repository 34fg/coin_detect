import cv2
from utils import imgProcessing, findContours

# Open a connection to the cam
cap = cv2.VideoCapture(1)

while True:
    # Read a frame from the cam
    ret, frame = cap.read()

    img_out= imgProcessing(frame)
    imgContours,conFound=findContours(frame,img_out,minArea=20)
    if conFound:
        for contour in conFound:
         peri = cv2.arcLength(contour ['cnt'], True)
         approx = cv2.approxPolyDP (contour ['cnt'], 0.02 * peri, True)
         if len(approx)>5:
            print(contour['area'])
    # else:
    #     print ('area')


    # Display the frame in a window
    cv2.imshow("Detected Coins", imgContours)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()