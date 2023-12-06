import cv2
import mediapipe as mp
import os
import uuid

def take_input():
#taking the roll number to store
    roll_number = input("Please enter the roll Number : ").capitalize()
    # Initialize MediaPipe Face Detection
    face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence = 0.5)


    path = os.path.join("data",roll_number)
    if(not os.path.exists(path)):
        os.makedirs(path)

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    i = 0
    while cap.isOpened() and i<20:

        ret, frame = cap.read()
        if not ret:
            continue

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Face Detection
        results = face_detection.process(rgb_frame)
        
        if results.detections:
            for detection in results.detections:
                
                #Getting the shape
                height, width, dep = frame.shape

                bboxC = detection.location_data.relative_bounding_box

                x, y, w, h = int(bboxC.xmin * width), int(bboxC.ymin * height), int(bboxC.width * width), int(bboxC.height * height)
                # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255))
                if cv2.waitKey(10) == ord('a'):
                    i = i+1
                    image = frame[y-10:y+h+10,x-10:x+w+10]
                    image = cv2.resize(image,(105,105))
                    img_path = os.path.join(path,'{}.jpg'.format(uuid.uuid4()))
                    cv2.imwrite(img_path,image)

        cv2.imshow(" ",frame)

        if cv2.waitKey(10) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
