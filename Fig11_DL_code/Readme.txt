Step 1 : Modify the config.txt to modify the path of training/testing dataset (1-10 is the number of the dataset)
Step 2 : Modify the file_dir in class TextConfig() in fastText.py/rcnn.py/text_cnn.py
Step 3 : Excute train.py to get the model of fasttext/RCNN/CNN
Step 4 : Excute eval.py to get the prediction of corresponding testing dataset


TextCNN folder records the code to train and test TextCNN model on our dataset
TextRCNN folder records the code to train and test TextRCNN model on our dataset
FastText folder records the code to train and test FastText model on our dataset

Training_dataset folder records the dataset used in Fig. 11
vocab folder records the word vector trained by Word2vec