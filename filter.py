import cv2 as cv
import dlib as dl
import numpy as np 

haar_cascade = cv.CascadeClassifier('haar_face.xml')
predictor = dl.shape_predictor('shape_predictor_68_face_landmarks.dat')

ronaldo_img = cv.imread("ronaldo.jpeg")
ronaldo_gray = cv.cvtColor(ronaldo_img,cv.COLOR_BGR2GRAY)

ronaldo_face = haar_cascade.detectMultiScale(ronaldo_gray,scaleFactor=1.1,minNeighbors=6)
x,y,w,h = ronaldo_face[0]
rect = dl.rectangle(x,y,x+w,y+h)
ronaldo_landmarks = predictor(ronaldo_gray,rect)

ronaldo_landmarks_array = np.array([[p.x,p.y] for p in ronaldo_landmarks.parts()])

def apply_filter(frame):
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)

    for  (x,y,w,h) in faces : 
        rect = dl.rectangle(x,y,x+w,y+h)
        faces_landmarks = predictor(gray,rect)
        points = np.array([[p.x,p.y] for p in faces_landmarks.parts()])

        h,status = cv.findHomography(ronaldo_landmarks_array,points)
        warped_img = cv.warpPerspective(ronaldo_img,h,(frame.shape[1],frame.shape[0]))

        mask = np.zeros_like(gray)
        cv.fillConvexPoly(mask,points,255)
        warped_mask = cv.warpPerspective(mask,h,(frame.shape[1],frame.shape[0]))
        warped_mask = cv.cvtColor(warped_mask,cv.COLOR_GRAY2BGR)

        frame = cv.addWeighted(frame,1.0,warped_img,0.7,0)
        frame[warped_mask > 0] = warped_img[warped_mask > 0]

    return frame

# for live webcam : 
cap = cv.VideoCapture(0)

# for sample video : 
# video_path = 'sample_video.mp4'
# cap = cv.VideoCapture(video_path)


while (True):
    ret, frame = cap.read()
    if not ret : 
        break

    frame = apply_filter(frame)
    cv.imshow('deep fake filter',frame)
    # if cv.waitKey(1) and 0xFF == ord('a'):
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    


cap.release()
cv.destroyAllWindows()


