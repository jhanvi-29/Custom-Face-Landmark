import tensorflow as tf
import numpy as np
import dlib
import cv2
import pandas as pd
import os

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def Landmark_to_CSV(image_name):
    image=cv2.imread(image_name)
    
    landmarks_x = []
    landmarks_y = []
    # print(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect the face
    rects = detector(gray, 1)
    # Detect landmarks for each face
    for rect in rects:
        # Get the landmark points
        shape = predictor(gray, rect)
        # Convert it to the NumPy Array
        shape_np = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            landmarks_x.append(shape.part(i).x)
            landmarks_y.append(shape.part(i).y)
        shape = shape_np

        # Make some example data
        x = landmarks_x
        y = landmarks_y

        # Create a figure. Equal aspect so circles look circular
        fig,ax = plt.subplots(1)
        ax.set_aspect('equal')

        # Show the image
        ax.imshow(image)

        # Now, loop through coord arrays, and create a circle at each x,y pair
        for xx,yy in zip(x,y):
            circ = Circle((xx,yy),1,color='Red')
            ax.add_patch(circ)

        # Show the image
        plt.show()
        fig.savefig('plot.png')
#         cv2.imwrite('output_face.jpg',image)
    all_landmarks = []
    filename = image_name
    all_landmarks.append(filename)
    for i,j in zip(landmarks_x,landmarks_y):

        all_landmarks.append(i) 
        all_landmarks.append(j)

    # append all landmarks in csv    
    l = [all_landmarks]
    with open('newdata.csv', 'a') as f: 
        write = csv.writer(f)
        write.writerows(l)
        
    return all_landmarks




path = "/home/jhanvipatel/Desktop/python/CustomFaceLandmark-20221220T090410Z-001/CustomFaceLandmark/images/"

for x in os.listdir(path):
    if x.endswith(".jpg"):
        Landmark_to_CSV(x)
