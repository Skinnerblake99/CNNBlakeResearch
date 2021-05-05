import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50 
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt3
import matplotlib.pyplot as plt4

##Enforce GPU useage
import tensorflow as tf
import tensorflow.keras

#Prediction imports
import pandas as pd
import tqdm as tqdm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from PIL import Image

##This if we have a GPU installed allows growth of model to avoid
##CUBLAS_STATYS_ALLOC_FAILED
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUd")
    except RuntimeError as e:
        print(e)


##model stuff                                                

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow import keras

Model = keras.models.load_model('D:/University/Year3/FYP assigment/NewModelArea/my_model.h5')
##Load local trained model

#Save as dot_img_file of the model
dot_img_file = '/tmp/model_1.png'
tf.keras.utils.plot_model(
    Model,
    to_file="model.png",
    show_shapes=True,
    show_dtype=True,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    ##DPI used to be 96
    dpi=192,
)



#Prediction for model 
test_df = pd.read_csv("Test2.csv")
test_image = []
for i in tqdm.tqdm(range(test_df.shape[0])):
    img = image.load_img(test_df["Path"][i], target_size=(224,224,3))
    img = image.img_to_array(img)
    #Changed to get batch size
    batch_size = 32
    img = img/255
    test_image.append(img)
X_test = np.array(test_image)


##21/03/2021 added here

predict_datagen = ImageDataGenerator(rescale=1./22.,
                                    #rotation_range=40,
                                    #width_shift_range=0.2,
                                    #height_shift_range=0.2,
                                    #shear_range=0.2,
                                    #zoom_range=0.2,
                                    #horizontal_flip=True,
                                    #fill_mode="nearest",
                                    preprocessing_function=preprocess_input)

predict_generator = predict_datagen.flow_from_directory(
                                                        'Dataset/Test',
                                                        target_size=(224,224),
                                                        color_mode="rgb",
                                                        batch_size=(32),
                                                        class_mode="categorical",
                                                        shuffle=False)



test_df.head()
test_paths = test_df["Path"]
y_test = test_df["Class"].values
y_test = to_categorical(y_test)


probabilities = Model.predict_classes(predict_generator,
                                      #batch_size=None,
                                      verbose=0,
                                      #steps=None,
                                      #callbacks=None,
                                      #max_queue_size=10,
                                      #workers=1,
                                      #use_multiprocessing=False
                                      )


from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
y_true = np.array([1] * 194 + [0] * 194)
x_pred = probabilities > 0.5
x_pred = x_pred.astype(int)
print (y_test.shape)
print (x_pred.shape)
x_pred = x_pred.reshape(-1,1)
y_test = y_true.reshape(-1,1)
print (y_test.shape)
print (x_pred.shape)
clf= SVC(random_state=0)
clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(x_pred,y_test.ravel())
plot_confusion_matrix(clf,
x_pred,
y_test,
normalize='all')
plt2.show()

##Fscore outputted here as well as precision and recall
fScore = f1_score(y_test,x_pred,labels=None, pos_label=1,average='binary',sample_weight=None,zero_division='warn')
precison = precision_score(y_test, x_pred,labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
recall = recall_score(y_test, x_pred,labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
print("fScore = :",fScore)
print("Precison = :",precison)
print("Recall = :",recall)

#metrics.accuracy_score(y_true,y_pred)
#metrics.multilabel_confusion_matrix(y_true,y_pred) 

y_test = test_df["Class"].values
y_test = to_categorical(y_test)


pred_df = pd.DataFrame()
pred_df["Img_Path"] = test_paths
pred_df["Prediction"] = probabilities
pred_df.to_csv("pred.csv", header=True, index=False)
pred = pd.read_csv("pred.csv")
##Image we want to make prediction for under here displaying
img_num = 15
img = plt3.imread(test_df["Path"][img_num])
plt3.imshow(img)
plt3.figtext(0,0,pred["Prediction"][img_num])
plt3.suptitle(test_df["Class"][img_num]) 
plt3.show()
print("The actual class is :", test_df["Class"][img_num])
print("The predicted class is:", pred["Prediction"][img_num]) 