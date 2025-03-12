import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis


app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

def extract_facial_features(image):

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    

    faces = app.get(image_rgb)
    
    feature_matrices = []
    
    for face in faces:
        landmarks_3d = face.landmark_3d_68 
        

        features = np.array(landmarks_3d)
        feature_matrices.append(features)


        for x, y, z in features.astype(int):
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    
    return np.array(feature_matrices), image, len(faces)


image_path = input("Enter the path to the image: ")
image = cv2.imread(image_path)
if image is None:
    print("Error: Could not read the image.")
    exit()


features, image_with_landmarks, num_faces = extract_facial_features(image)


cv2.imshow("Detected Faces with Landmarks", image_with_landmarks)
cv2.waitKey(0)
cv2.destroyAllWindows()

if num_faces > 0:
    np.save("facial_features_retinaface.npy", features)
    print("Facial features saved as 'facial_features_retinaface.npy'")
    print(f"Total Faces Detected: {num_faces}")
    print("Feature Matrices (3D landmarks):")
    for idx, matrix in enumerate(features):
        print(f"Matrix {idx + 1} Data:")
        print(matrix)
else:
    print("No face detected.")
