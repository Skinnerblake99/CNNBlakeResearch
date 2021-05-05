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

##Augment here
train_datagen = ImageDataGenerator(rescale=1./22.,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode="nearest",
                                   preprocessing_function=preprocess_input)

train_generator=train_datagen.flow_from_directory("Dataset/Train",
                                                  target_size=(224,224),
                                                  color_mode="rgb",
                                                  batch_size=32,
                                                  class_mode="categorical")
##No need to augment the validation material 

valid_datagen = ImageDataGenerator(rescale=1./22.,
                                   preprocessing_function=preprocess_input)
valid_generator=valid_datagen.flow_from_directory("Dataset/Valid",
                                                  target_size=(224,224),
                                                  color_mode="rgb",
                                                  batch_size=32,
                                                  class_mode="categorical")

##model stuff                                                

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Flatten

base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
Model = Sequential()
Model.add(base_model)
Model.add(Flatten())
Model.add(Dense(5292,activation="relu"))
Model.add(Dense(4200,activation="relu"))
Model.add(Dense(3400,activation="relu"))
Model.add(Dropout(0.5))
Model.add(Dense(2646,activation="relu"))
Model.add(Dropout(0.5))
Model.add(Dense(1323,activation="relu"))
Model.add(Dense(512,activation="relu"))
Model.add(Dropout(0.5))
Model.add(Dense(256,activation="relu"))
Model.add(Dense(128,activation="relu"))
Model.add(Dropout(0.5))
Model.add(Dense(64,activation="relu"))
Model.add(Dense(32,activation="relu"))
Model.add(Dense(16,activation="relu"))
Model.add(Dense(2, activation="softmax"))


base_model.trainable = False

##Optimising and learning rate here 
from tensorflow.keras import optimizers

opt = optimizers.Adam(lr=0.001)

Model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

step_size_train = train_generator.n//train_generator.batch_size
step_size_valid = valid_generator.n//valid_generator.batch_size


history = Model.fit(
    train_generator,
    steps_per_epoch=step_size_train,
    epochs=100,
    validation_data=valid_generator,
    validation_steps=step_size_valid)

##Save the model
Model.save('D:/University/Year3/FYP assigment/NewModelArea/my_model.h5')
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
    dpi=192,
)

##Training and validation loss graph
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(0,100)
plt4.plot(epochs, loss_train,'g', label= 'Training loss')
plt4.plot(epochs, loss_val,'b', label= 'Validation loss')
plt4.title('Training and Validation Loss Over Epochs')
plt4.xlabel('Epochs')
plt4.ylabel('Loss')
plt4.legend()
plt4.show()

#Training and Validation Accurarcy
loss_train = history.history['accuracy']
loss_val = history.history['val_accuracy']
epochs = range(0,100)
plt.plot(epochs,loss_train,'g', label = 'Training accuracy')
plt.plot(epochs,loss_val,'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

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

##Fscore  , Precision and Recall outputted here
fScore = f1_score(y_test,x_pred,labels=None, pos_label=1,average='binary',sample_weight=None,zero_division='warn')
precison = precision_score(y_test, x_pred,labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
recall = recall_score(y_test, x_pred,labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
print("Fscore = :",fScore)
print("Precision = :",precison)
print("Recall = :",recall)
y_test = test_df["Class"].values
y_test = to_categorical(y_test)


pred_df = pd.DataFrame()
pred_df["Img_Path"] = test_paths
pred_df["Prediction"] = probabilities
pred_df.to_csv("pred.csv", header=True, index=False)
pred = pd.read_csv("pred.csv")
##Input Image number to read from to make prediction against
img_num = 15
img = plt3.imread(test_df["Path"][img_num])
plt3.imshow(img)
plt3.figtext(0,0,pred["Prediction"][img_num])
plt3.suptitle(test_df["Class"][img_num]) 
plt3.show()
print("The actual class is :", test_df["Class"][img_num])
print("The predicted class is:", pred["Prediction"][img_num]) 



