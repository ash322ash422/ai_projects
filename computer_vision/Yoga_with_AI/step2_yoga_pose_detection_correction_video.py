import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
import pickle

video_path = 'downward_dog.mp4' 
pose_classifier_path = "pose_classifier.p"

def pose_correction(landmarks, mp_pose):#TODO More condition can be added here
    correct_pose=""
    if landmarks:
        # Get the angle between the left hip, knee and ankle points. 
        left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
        
        # Get the angle between the right hip, knee and ankle points 
        right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
        if all(190 < angle < 250 for angle in (left_knee_angle, right_knee_angle)): 
            correct_pose = "Move your knee cap up"
    
    return correct_pose   
    
    
def calculateAngle(landmark1, landmark2, landmark3):
    '''
    This function calculates angle between three different landmarks.
    Args:
        landmark1: The first landmark containing the x,y and z coordinates.
        landmark2: The second landmark containing the x,y and z coordinates.
        landmark3: The third landmark containing the x,y and z coordinates.
    Returns:
        angle: The calculated angle between the three landmarks.
    '''
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    # Check if the angle is less than zero.
    if angle < 0:
        # Add 360 to the found angle.
        angle += 360
    return angle

def load_classifier():
    # Load the trained classifier
    with open(pose_classifier_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model']

def predict_pose(image, classifier, pose_detector=mp.solutions.pose.Pose(static_image_mode=True), class_names=None):
    prediction_index = None ; predicted_class = None ; confidence = None
    # Predict pose for a given image
    img = image
    results = pose_detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
        print(f"No pose detected.")
        return prediction_index,predicted_class,confidence
    img_with_pose = img.copy()

    # Extract landmark data
    landmark_data = []
    for landmark in results.pose_landmarks.landmark:
        x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
        landmark_data.extend([landmark.x, landmark.y, landmark.z])

    # Make prediction
    prediction_index = classifier.predict([landmark_data])[0]
    predicted_class = class_names[prediction_index]
    confidence = np.max(classifier.predict_proba([landmark_data]))

    return prediction_index,predicted_class,confidence

def detectPose(image, pose, display=True):
    # Initializing mediapipe pose class.
    mp_pose = mp.solutions.pose

    # Initializing mediapipe drawing class, useful for annotation.
    mp_drawing = mp.solutions.drawing_utils 
    
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Pose Detection.
    results = pose.process(imageRGB)
    
    # Retrieve the height and width of the input image.
    height, width, _ = image.shape
    
    # Initialize a list to store the detected landmarks.
    landmarks = []
    
    # Check if any landmarks are detected.
    if results.pose_landmarks:
    
        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        
        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
    
    # Check if the original input image and the resultant image are specified to be displayed.
    if display:
    
        # Display the original input image and the resultant image.
        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
        # Also Plot the Pose landmarks in 3D.
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
    else:
        # Return the output image and the found landmarks.
        return output_image, landmarks
    
def detectPoseOnVideo(video=None):
    # Initializing mediapipe pose class.
    mp_pose = mp.solutions.pose

    # Setup Pose function for video.
    pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

    # Create named window for resizing purposes
    cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)

    # Set video camera size
    video.set(3,1280)
    video.set(4,960)

    # Initialize a variable to store the time of the previous frame.
    time1 = 0
    classifier = load_classifier()
    class_names = ['downdog', 'goddess', 'plank']
    # Iterate until the video is accessed successfully.
    while video.isOpened():
        # Read a frame.
        ok, frame = video.read()
        
        # Check if frame is not read properly.
        if not ok:
            break # Break the loop.
        
        # Flip the frame horizontally for natural (selfie-view) visualization.
        frame = cv2.flip(frame, 1)
        
        # Get the width and height of the frame
        frame_height, frame_width, _ =  frame.shape
        
        # Resize the frame while keeping the aspect ratio.
        frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
        
        # Perform Pose landmark detection.
        frame, landmarks = detectPose(image=frame, pose=pose_video, display=False)
        prediction_index,predicted_class,confidence = predict_pose(image=frame,
                                                                    classifier=classifier,
                                                                    pose_detector=mp.solutions.pose.Pose(static_image_mode=True),
                                                                    class_names=class_names
        )
        print(f"prediction_index : {prediction_index}, predicted_class: {predicted_class}, confidence:{confidence}")
        correct_pose=""
        if landmarks:
            correct_pose = pose_correction(landmarks=landmarks,mp_pose=mp_pose)
        
        # Set the time for this frame to the current time.
        time2 = time()
        
        # Check if the difference between the previous and this frame time > 0 to avoid division by zero.
        if (time2 - time1) > 0:
        
            # Calculate the number of frames per second.
            frames_per_second = 1.0 / (time2 - time1)
            
            # Write the calculated number of frames per second on the frame. 
            cv2.putText(frame, "Pose: {}, {}".format(predicted_class,correct_pose), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2,
                        (0, 255, 0), 3
            )
        
        # Update the previous frame time to this frame time.
        # As this frame will become previous frame in next iteration.
        time1 = time2
        
        # Display the frame.
        cv2.imshow('Pose Detection', frame)
        
        # Wait until a key is pressed.
        # Retreive the ASCII code of the key pressed
        k = cv2.waitKey(1) & 0xFF
        
        # Check if 'ESC' is pressed.
        if(k == 27):
            
            # Break the loop.
            break

    # Release the VideoCapture object.
    video.release()

    # Close the windows.
    cv2.destroyAllWindows()
    
#MAIN execution

# Initialize the VideoCapture object to read from a video stored in the disk.
video = cv2.VideoCapture(video_path)
detectPoseOnVideo(video=video)
        
    
