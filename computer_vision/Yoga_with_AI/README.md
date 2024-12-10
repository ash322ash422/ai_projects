# INSTRUCTIONS

I used python3.11. Package requirements for this project: mediapipe, scikit-learn. You can run this by downloading following 3 files: step2_yoga_pose_detection_correction_video.py, downward_dog.mp4 and pose_classifier.p The demo video is in result.mp4 

# MY THOUGHT PROCESS

# step1:

a) I downloaded the training/testing dataset from  https://www.kaggle.com/niharika41298. Then I cleaned the data and reduce to just using 3 yoga poses :'downdog', 'goddess', 'plank' . The dataset is in directory DATASET or if you cannot find it, then ,alternatively, you can download from above link and put it there.

b) I used 'mediapipe' to find landmarks (AKA as keypoints of body) and saved the landmarks data in separate file. Mediapipe is well suited for detecting keypoints in Yoga poses.

c) I used the model/classifier 'RandomForestClassifier', and employed GridSearchCV to find the best model corresponding to the best hyperparameter. I used confusion matrix to check the reults and I was satisfied. I saved the classifier. I also tried YOLO8, but RandomForestClassifier gave me better results.

All the above steps were done in python script 'step1_yoga_pose_training_classifier.ipynb'  (jupyter notebook). It was easy/convenient this way.

# step 2:

I loaded the classifier "pose_classifier.p" that was trained in above-mentioned step 1. I ran a 5-6 sec long sample video downward_dog.mp4 and performed detection on each frames for only one specific yoga position : downward dog position. Since I wanted to show proof of concept, so I used only one yoga pose (downward dog) and just one simple condition for correcting pose if it was not done properly: namely, the left_knee_angle and right_knee_angle. These 2 angles should be between 180 and 190 degrees, and if these angles deviate and become 190 < angle < 250, I display helpful message to correct the pose. I kept it simple for demo purpose.

# step 3:

The next step would be 

a) Train on diverse datasets for different Yoga poses. 

b) Add more conditions for error correction in body posture.

c) Employ better hardware for getting realtime output without any lag. Employ YOLO model, fine tune it and check which performs better. 

d) Optimize the code for faster response. If needed, use compiled language C++, java, etc.

e) To enhance user experience, provide feedback in the form of audio and video, with a little music playing in the background depending on the pose.

