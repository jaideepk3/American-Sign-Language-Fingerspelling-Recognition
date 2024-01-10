# American Sign Language Fingerspelling Recognition

This project aims to develop a machine learning model that can recognize American Sign Language (ASL) fingerspelling from video frames. Fingerspelling is a technique of using hand shapes to spell words or abbreviations. It is often used by deaf or hard of hearing people to communicate words that do not have a sign, such as names, places, or technical terms.

## Dataset

The dataset used for this project is from the [Google ASL Fingerspelling Recognition Competition](https://www.kaggle.com/competitions/asl-fingerspelling). It consists of over 200,000 frames of ASL fingerspelling video, with 24 classes corresponding to the letters A-Z (excluding J and Z, which require motion). The dataset also includes randomly generated addresses, phone numbers, and urls derived from components of real addresses/phone numbers/urls, as well as fingerspelled sentences.

The dataset provides the landmarks of the face, pose, and hands for each frame, extracted with the [MediaPipe holistic model](https://google.github.io/mediapipe/solutions/holistic.html). The landmarks are normalized by MediaPipe and stored in a Parquet format.

## Model

The model used for this project is a convolutional neural network (CNN) with a recurrent neural network (RNN) layer. The CNN layer extracts features from the landmarks of the hands, while the RNN layer captures the temporal information of the frames. 

The model is trained with categorical crossentropy loss and optimized with Adam optimizer. The model uses a learning rate scheduler to adjust the learning rate during training. The model also uses a top-k accuracy metric to evaluate the performance on the validation and test sets.

## Results

The model achieves a normalized Levenshtein distance (NLD) of 0.83 on the test set, which means that the average edit distance between the predicted and true labels is 17% of the average phrase length. The model also achieves a top-1 accuracy of 86.5% and a top-3 accuracy of 96.2% on the test set.

## References

- [TensorFlow ASLFR](https://www.kaggle.com/code/gusthema/asl-fingerspelling-recognition-w-tensorflow/notebook)
- [1st Place Solution - Training](https://www.kaggle.com/code/hoyso48/1st-place-solution-training)
- [1st Place Solution - Inference](https://www.kaggle.com/code/hoyso48/1st-place-solution-inference)
- [Fingerspelling Detection in ASL paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Shi_Fingerspelling_Detection_in_American_Sign_Language_CVPR_2021_paper.pdf)
- [ASLFR using Hybrid Deep Learning](https://www.irjet.net/archives/V10/i9/IRJET-V10I906.pdf)
- [TopKAccuracy](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/TopKCategoricalAccuracy)
- [Categorical Croosentropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy)
- [Learning Rate Scheduler](https://d2l.ai/chapter_optimization/lr-scheduler.html)
- [Levenshtein Distance](https://www.geeksforgeeks.org/introduction-to-levenshtein-distance/)
