import cv2
import face_recognition
import numpy as np
import csv
from datetime import datetime

video_capture= cv2.VideoCapture(0)

mark_image = face_recognition.load_image_file(r"D:\UNIVERSITY ASSIGNMENTS\SEMESTER 6\DIP FINAL PROJECT\Photos/mark.jpg")
mark_encoding = face_recognition.face_encodings(mark_image)[0]

steve_image = face_recognition.load_image_file(r"D:\UNIVERSITY ASSIGNMENTS\SEMESTER 6\DIP FINAL PROJECT\Photos/steve.jpg")
steve_encoding = face_recognition.face_encodings(steve_image)[0]

trump_image = face_recognition.load_image_file(r"D:\UNIVERSITY ASSIGNMENTS\SEMESTER 6\DIP FINAL PROJECT\Photos/trump.jpg")
trump_encoding = face_recognition.face_encodings(trump_image)[0]

elon_image = face_recognition.load_image_file(r"D:\UNIVERSITY ASSIGNMENTS\SEMESTER 6\DIP FINAL PROJECT\Photos/elon.jpg")
elon_encoding = face_recognition.face_encodings(elon_image)[0]

obama_image = face_recognition.load_image_file(r"D:\UNIVERSITY ASSIGNMENTS\SEMESTER 6\DIP FINAL PROJECT\Photos/obama.jpg")
obama_encoding = face_recognition.face_encodings(obama_image)[0]

ali_image = face_recognition.load_image_file(r"D:\UNIVERSITY ASSIGNMENTS\SEMESTER 6\DIP FINAL PROJECT\Photos/ali.jpg")
ali_encoding = face_recognition.face_encodings(ali_image)[0]

sudais_image = face_recognition.load_image_file(r"D:\UNIVERSITY ASSIGNMENTS\SEMESTER 6\DIP FINAL PROJECT\Photos/sudais.jpg")
sudais_encoding = face_recognition.face_encodings(sudais_image)[0]

known_face_encoding = [mark_encoding,steve_encoding,trump_encoding,elon_encoding,obama_encoding,ali_encoding,sudais_encoding]

known_faces_names = ["mark","steve","trump","elon","obama","ali","sudais"]

students = known_faces_names.copy()

face_locations= []
face_encodings= []
face_names= []
s = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date+'.csv','w+',newline='')
Inwriter = csv.writer(f)

while True:

    _,frame = video_capture.read()

    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)

    rgb_small_frame = small_frame[:,:,::-1]

    if s:

        face_locations = face_recognition.face_locations(rgb_small_frame)

        face_encodings=face_recognition.face_encodings(rgb_small_frame,face_locations)

        face_names = []

        for face_encoding in face_encodings:

            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)

            name=""

            face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)

            best_match_index = np.argmin(face_distance)

            if matches[best_match_index]:

                name = known_faces_names[best_match_index]
                
            face_names.append(name)

            if name in known_faces_names:

                if name in students:

                    students.remove(name)

                    print(students)

                    current_time = now.strftime("%H-%M-%S")

                    Inwriter.writerow([name,current_time])

    cv2.imshow("attendence management system",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
                  
video_capture.release()
cv2.destroyAllWindows()
f.close()
       
           
              
         
   
    
                  
            

    
    
    
    
    
    

                                             
                    



