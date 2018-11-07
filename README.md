# Emotion_Recognition
Emotion Recognition model using Convolution Neural Network . Model accurarcy is not upto the mark. I dont have powerful GPUs . So I am unable to built complex model. If can then you are most welcome.

**Dataset**

[FER-2013 Faces Database](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

Number of images for training= 28,709

Number of images for Testing= 3,589

Image format=(48x48) in grayscale

No of Emotions=7 (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)

**Training**

file name- *ER_model_train.py*

library used- keras, tensorflow

**Model**

file name- *ER_model_training12ep.h5*

no of hidden layer= 3

pooling used- Max and Average

dropout=0.5

Activation -relu



**Prediction**

file name- *ER_model_predict.py*

to predict enter the image file name along with its location
