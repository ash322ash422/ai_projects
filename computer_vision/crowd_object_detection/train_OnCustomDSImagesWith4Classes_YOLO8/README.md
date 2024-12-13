- Used python3.9 and installed ultralytics, scikit-learn

- This dataset was taken from https://datasetninja.com/labeled-surgical-tools-and-images

- The goal was to address object detection using YOLO8. YOLO8 has 80 classes, whereas the present dataset had only 4 classes. So I had to fine tune the YOLO8. The present dataset comprises a total of 1834 training images, each accompanied by corresponding labels. These labels categorize the objects into four types: scalpel nº4, straight dissection clamp, straight mayo scissor, or curved mayo scissor. 

##########################################
# STEPS:
1) Date collection: First I downloaded the dataset directly from https://datasetninja.com/labeled-surgical-tools-and-images. The dataset was divided into 2 parts: train(1834) and test(786). Each dir had 2 sub directories: ann and images. 

2) Data preprocessing: This dataset was not compatible with YOLO8 requirement for dataset format.
 - First, I reformatted this dataset into YOLO8 format and copied into different location.
 - Second, due to limitation in  computation  resource, I extracted small data records from dir 'train' and used 80% of those for training and 20% for validation.
    I extracted test dataset, which was arbitrarily choosen as 10% of total data, from dir 'test' for testing.

3)Train and Save: I used Google Colab to train and save the model. I uploaded the dataset on colab, then I downloaded a pre-trained nano YOLO8 model 'yolov8n.pt',
 and fine tuned the model on my custom  dataset. I downloaded the saved model on local desktop. 
 I also downloaded the dir 'runs' in zipped format which stores the results. Later, I unzipped it into dir 'detect'

4) Load and predict: On desktop, I loaded the above saved model and tested the prediction. 

RUN1: 
- Here I used 300 data rec. The training size and validation size was 300*.8=240 and 300-240=60, respect.
- The testing size was 30.
- The number of epocs was 15

RUN2:
- Here I increased dataset to 400 data rec. The training size and validation size was 400*.8=320 and 400-320=80, respect.
- The testing size was 30
- The number of epocs was increased to 30

Analysis: By observing the image of normalized confusion matrix located in dir detect_runX/train/confusion_matrix_normalized.png, I note following:
The accuracy for 'curved mayo scissor'       went up from 38% to 45%
The accuracy for 'straight mayo scissor'     went up from 55% to 71%
The accuracy for 'scalpel n\u00ba4'          went up from 67% to 88%
The accuracy for 'straight dissection clamp' went up from 50% to 73%


Conclusion:
Accuracy can be improved further by taking the whole dataset, augmenting the dataset, changing batch size and increasing the number of epochs.
