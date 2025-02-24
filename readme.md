# Get the training dataset

[Google drive link to the dataset](https://drive.google.com/open?id=1V8rQ5QR5q5zn1NHJhhf-6xIeDdXVtYs9)

## Reformatting the dataset

- The original data set has to be renamed to Action-Recognition-Dataset and moved to the project's root folder

- The folders inside the Action-Recognition-Dataset/source-images3 have to be renamed to jump_1, jump_2, kick_1, ...etc

![Folder Structure Image](ExtraSrc(unimportant)/readmeFolderStructure.png)

### COMPLETE THE ABOVE STEPS AND CONTINUE


#### There are 4 folders for "stand" but in valid_images.txt there are 5 entries for stand. stand3 has been repeated twice:

### prev:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;after:

stand_1 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;stand_1   
55 520 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 55 520

stand_2  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;stand_2   
60 500 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;60 500

stand_3 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;stand_3   
35 395 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;35 395   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;481 619

stand_4  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;stand_4   
481 619 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;40 335

stand_5  
40 335


### AND

walk_1 and walk_2 has to be interchanged

# Preprocessing the data
### Run the [getSkelsFromImg.py](src/getSkelsFromImg.py) to get the co-ordinates of the 33 points of the skeleton of all the valid images of the dataset
### The resultant csv files will be created in the directory [raw_skels](raw_skels)

# Training the model
### You can experiment with changing the hyperparameters of the model by changing the parameters in MLPClassifier
### Run the [train.py](src/train.py) to train the model

# Test the accuracy of the model
### To test the accuracy of the model run [accuracy.py](src/accuracy.py)

# To Classify an unseen image 
### Create and upload a image in testImgs folder and run [predict.py](src/predict.py)
### The result will be printed on the console