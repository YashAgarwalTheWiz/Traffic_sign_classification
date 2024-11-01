The Traffic Sign Classification System aims to develop a robust model that accurately identifies various traffic signs using computer vision techniques. By leveraging deep learning, specifically Convolutional Neural Networks (CNNs), the system can classify images of traffic signs in real-time, enhancing road safety and assisting autonomous vehicles.

Objectives
Classification: To classify different traffic signs based on images captured from various environments.
Real-Time Prediction: To enable real-time identification of traffic signs in a user-friendly web interface.
User Interaction: To allow users to upload images and receive immediate feedback on the type of traffic sign displayed.
Dataset
The project utilizes the German Traffic Sign Recognition Benchmark (GTSRB) dataset, which contains over 50,000 images of traffic signs categorized into 43 classes. Each class represents a specific traffic sign, such as speed limits, warnings, and regulatory signs.

Methodology
Data Preprocessing:
Images are resized to a uniform dimension of 50x50 pixels.
Pixel values are normalized to a range between 0 and 1.
The dataset is split into training and validation sets.

Model Architecture:
A Sequential CNN model is built with several convolutional layers, max-pooling layers, and dropout layers to prevent overfitting.
The output layer employs a softmax activation function to handle multi-class classification.

Training:
The model is trained using the training set for 15 epochs, with a batch size of 128.
Loss is calculated using sparse categorical cross-entropy, and the Adam optimizer is used for efficient training.

Evaluation:
The model's performance is assessed on the validation set, and accuracy is visualized through loss and accuracy plots.

Deployment:
A web interface is created using Streamlit, allowing users to upload images of traffic signs.
The model processes the uploaded images, predicts the class of the traffic sign, and displays the result.

Features
Image Upload: Users can upload images in JPG or PNG format.
Real-Time Classification: The model predicts the traffic sign class based on the uploaded image.
User-Friendly Interface: A simple interface to enhance user experience.
