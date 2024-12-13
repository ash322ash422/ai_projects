#This script is used to download a model and store it on local machine. I used python3.11 virtual environment
# step 1) >  pip install tensorflow segmentation-models

import tensorflow as tf

# Load a pre-trained segmentation model from tf.keras.applications
model = tf.keras.applications.MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')

# Add custom layers for segmentation
x = model.output
x = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(x)  # 1 class for segmentation
x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)  # Up-sample to original size

# Create the final model
segmentation_model = tf.keras.models.Model(inputs=model.input, outputs=x)

# Save the model
segmentation_model.save('segmentation_model.h5')
print("Model saved as segmentation_model.h5")
