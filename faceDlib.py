import cv2
import dlib
import numpy as np
import pandas as pd

face_detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"
shape_predictor = dlib.shape_predictor(predictor_path)

def extract_facial_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    
    feature_matrices = []
    
    for face in faces:
        landmarks = shape_predictor(gray, face)
        features = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)])
        feature_matrices.append(features)

        for i in range(68):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    
    return np.array(feature_matrices), image, len(faces)


image_path = input("Enter the path to the image: ")
image = cv2.imread(image_path)
if image is None:
    print("Error: Could not read the image.")
    exit()

features, image_with_landmarks, num_faces = extract_facial_features(image)

cv2.imshow("Detected Faces", image_with_landmarks)
cv2.waitKey(0)
cv2.destroyAllWindows()

if num_faces > 0:
    print(f"Total Faces Detected: {num_faces}")
    print("Feature Matrices:")
    for idx, matrix in enumerate(features):
        print(f"Matrix {idx + 1} Data:")
        print(matrix)
        print()
    
    np.save("facial_features.npy", features)  # Save as a NumPy file
    print("Facial features saved to 'facial_features.npy'")
else:
    print("No face detected.")