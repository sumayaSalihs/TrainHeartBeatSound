#  This is ML Training was inpired and directed by this project: https://www.kaggle.com/code/mohammadrezashafie/par-ebi
#  Train Heart/Lung Sound Model One ~ Mboalab ~ Outreachy 2023


# Imports all libraies needed to run this project
import glob
import os
import librosa as lib
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline/
import librosa.display
import numpy as np
import IPython.display as ipd
import shutil
import soundfile
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
#import shutil
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
#from sklearn.metrics import accuracy_score,
#from mlxtend.plotting import plot_confusion_matrix
#from tensorflow.contrib import lite
import time
import tensorflow as tf





class TrainHeartBeatData:
    """
    A class used to train heartbeats
    
    ...
    
    Attributes
    ----------
    
    
    
    Methods
    -------
    get_fileNames(path)
                Extracts the file names from the audio files
    create_dataframe(dataframe_name)
                Creating dataframe from the labelled audio files having duration equal to and more than 3 seconds
    label_count()
                Counts the counts of each labels in the dataframe
    data_distribution(data)
                Plots Data Category Distribution
    spectrogram(file_path, label)
               Plotting spectogram of an audio signal
    get_raw_data()
               Get raw data file formats in a list form
    get_unique_data_label()
               Get the unique labels in the raw dataset
    waveform(file_path, label)
               Plotting Waveform of an audio signal
    play_raw_audio_File(file_path)
               Plays raw audio file
    changing_speed(speed_rate, src_path, dst_path)          
               Changing only the speed of an audio signal with different rates and saving it
    changing_pitch(step, src_path, dst_path)       
               Changing only the pitch of an audio signal with different steps and saving it
    sound_augmentation(src_path, dst_path)
               Creating new files from changing pitch and speed of the input audio files
    create_data2_folder()
               Creates folder named Data2 where copied files from both set_a and set_b sits 
    fill_folder1_toData2(self, folder_name)
               fills Data2 folder with copies of sound data from both set_a and set_b folder
    chdir_to_working()
                Changes to directory where working is going to take place, i.e. data augmentation and training
    create_new_augmented_data_files()
                Checks and creats new directory for saving newly generated audio files using data augmentation
    create_dataframe_final(dataframe_name)
                Creating dataframe from the labeled audio files having duration equal to and more than 3 seconds, dataframe of final_data(old raw data and augmented file copies of it)
    split_for_training_testing(final_data)
                Splits the Data into Training and Testing Data in the proportion of 80:20 (Train:Test)
    feature_extraction(file_path)
                Sets the sampling audio rate to 22050, setting the duration only to 3 seconds and finally
    create_xtrain_xtest(train_data, test_data)
                Cretates x_train and x_test file which are feature extractions
    reshape_xtrain_x_test(x_train, x_test)
                Converts x_train and x_test data into shape appropriate for training or use with CNN
    encode_label(train_data, test_data)
                Encodes label, i.e. converts it to its numerical format which can be used for training
    y_train_y_test_to_categorical(y_train, y_test)
                Converts numerical label vector format to its binary format which is needed for training
    CNN_model(n_width,n_height,n_channels,n_dropout,n_classes)
                Creates a CNN model
    cnn_model(self, x_train)
                Creates cnn_model with required parameters
    setting_learning_rate_loss_ftn(cnn_model)
                Sets the learning rate and loss function for the model
    train_model(cnn_model, y_train, y_test, x_train, x_test)
                Trains model with all required parameters
    display_best_training_accruracy_loss_score(cnn_model, x_train, y_train)
                Gets the best training and loss score of the model
    display_best_testing_accruracy_loss_score(cnn_model, x_test, y_test)
                Gets the best accuracy and loss score of testing for the model
    calculate_display_precision(cnn_model, x_test, y_test)
                Calculates and Gets the model's precision
    get_cnn_history(cnn_history)
                Gets model history
    plot_epoch_vs_train_test_accuracy_graph(cnn_history_history)
                Gets graph of train and test accuracy from model history
    plot_epoch_vs_train_test_loss_graph(cnn_history_history)
                Gets graph of train and test loss score from model history
    confusion_matrix(y_actual,y_pred)
                Gets model's confusion matrix, returns graph plot for it
    get_restored_model_accuracy(cnn_model_value, x_test, y_test)
                Gets models accuracy and percentage
    save_model(cnn_model_value, file_name)
                Saves model in desired file format
    get_model_from_file(file_path)
                Gets module in tersorflow format from saved model file
    convert_model_to_tflite(file_name)
                Converts model to tflite format
    run_training_model()
                preprocesses data, train model and review model performance, can customize to suit users goal for training
    
    
    
    
    
    
    
    """
    
    data_path = "[Path_to_home_directory]/Desktop/HeartBeatSounds/archive/"
    parent_data_path = "[Path_to_home_directory]/Desktop/HeartBeatSounds/"
    encode = LabelEncoder() #used by several instances, You need to call fit(...) or fit_transform(...) on your LabelEncoder before you can access classes_
    
    
    def __init__(self):
        """
        Parameters
        ----------
        num_of_classes : int
                       The number of possible classifiers for a heartbeat
        sample_size : Dictionary
                      The key and value pair for classifier and its sample size
                     
        
        """
                      
                      
        #self.num_of_classes = num_of_classes
        #self.sample_size = sample_size
        
        
   
    def get_fileNames(self, path):
       fileNames = [file for file in glob.glob(path)]
       return fileNames
   
   
    
    def create_dataframe(self, dataframe_name):
       audio = {'file_path': [], 'label': []}
       unlabeled_files = ['Aunlabelledtest', 'Bunlabelledtest']
       for folder in [ self.data_path +'set_a/', self.data_path +'set_b/']:
           fileNames = self.get_fileNames(folder + '//**')
           for file in fileNames:
              label = os.path.basename(file).split('_')[0]
              if((lib.get_duration(filename=file)>3) and (label not in unlabeled_files)):
                 audio['file_path'].append(file)
                 audio['label'].append(label)
 
       dataframe_name = pd.DataFrame(audio)
       return dataframe_name
   
    
    
    def get_raw_data(self):
        return self.create_dataframe('raw_data')
    
    
    
    def get_unique_data_label(self):
        return self.get_raw_data().label.unique()
    
   
    
    def label_count(self):
        raw_data = self.get_raw_data
        return raw_data.label.value_counts()
    

    
    def data_distribution(self, data):
        plt.figure(figsize=(16,3))
        data.label.value_counts().plot(kind='bar', title="Data Category distribution")
        plt.show()
        
        
        
    def spectrogram(self, file_path, label):
      y, sr = lib.load(file_path)
      plt.figure(figsize=(16,3))
      plt.title(label + 'Log-Frequency Power Spectrogram')
      data = lib.amplitude_to_db(np.abs(lib.stft(y)), ref=np.max)
      lib.display.specshow(data, y_axis='log', x_axis='time')
      plt.colorbar();
      
      
      
    def waveform(self, file_path, label):
      y, sr = lib.load(file_path)
      plt.figure(figsize=(16, 3))
      plt.title(label + ' Sound Wave')
      librosa.display.waveplot(y)
      # ibrosa.display.waveplot(y, sr)


    def play_raw_audio_File(self, file_path):
        return ipd.Audio(file_path) #to hear sound play in Notebooks not interactive shell like IPython
    
    
    
    def changing_speed(self, speed_rate, src_path, dst_path):
        files = self.get_fileNames(src_path + "//**")
        if not os.path.exists(dst_path):
          os.makedirs(dst_path)
        for file in tqdm(files):
          label = os.path.basename(file).split('.')[0]
          y, sr = lib.load(file)
          updated_y = lib.effects.time_stretch(y, rate=speed_rate)
          soundfile.write(dst_path + '//' + label + '_' + str(speed_rate) + ".wav", updated_y, sr)
          
          
    def changing_pitch(self, step, src_path, dst_path):
        files = self.get_fileNames(src_path + '//**')
        if not os.path.exists(dst_path):
          os.makedirs(dst_path)
        for file in tqdm(files):
          label = os.path.basename(file).split('.')[0]
          y, sr = lib.load(file)
          updated_y = lib.effects.pitch_shift(y, sr, n_steps=step)
          soundfile.write(dst_path + '//' + label + '_' + str(step) + '.wav', updated_y, sr)
          
          
    
    def sound_augmentation(self, src_path, dst_path):
        speed_rates = [1.08, 0.8, 1.10, 0.9]
        for speed_rate in speed_rates:
            self.changing_speed(speed_rate, src_path, dst_path)
    
    
        steps = [2, -2, 2.5, -2.5]
        for step in steps:
            self.changing_pitch(step, src_path, dst_path)
    
        files = self.get_fileNames(src_path + '//**')
        for f in files:
          shutil.copy(f, dst_path)
          
      
    def create_data2_folder(self):
         os.chdir(self.data_path+'working/')
         os.mkdir('Data2')
        
        
    def fill_folder1_toData2(self, folder_name): #folder1: set_a , folder2: set_b
        source = self.data_path+folder_name
        destination = self.data_path+'working/Data2/'
        
        # Get a list of files in the source directory
        file_list = os.listdir(source)

        # Iterate over the files and copy them to the destination directory
        for file_name in file_list:
            source_file = os.path.join(source, file_name)
            destination_file = os.path.join(destination, file_name)
            shutil.copy2(source_file, destination_file)
            
            
            
    def chdir_to_working(self): #change directory and create a new directory there
        os.chdir(self.data_path+'working/')
        os.mkdir('OUT')
        
        
        
    def create_new_augmented_data_files(self):
        # Checking and creating new directory for saving newly generated audio files using data augmentation
        if os.path.exists(self.data_path+'working/OUT'):
          if len(self.get_fileNames(self.data_path+'working/OUT//**')) == 4175:
              print('Sound Augmentation Already Done and Saved')
          else:
              shutil.rmtree(self.data_path+'working/OUT')
              self.sound_augmentation(self.data_path+'working/Data2', self.data_path+'working/OUT')
        else:
            self.sound_augmentation(self.data_path+'working/Data2', self.data_path+'working/OUT')
        
            
            
            
    # Creating dataframe from the labeled audio files having duration equal to and more than 3 seconds
    def create_dataframe_final(self, dataframe_name):
        audio = {'file_path':[], 'label':[]}
        unlabeled_files = ['Aunlabelledtest', 'Bunlabelledtest']
        for folder in [self.data_path+'working/OUT/']:
            files = self.get_fileNames(folder + '//**')
            for file in files:
                label = os.path.basename(file).split('_')[0]
                if((lib.get_duration(filename=file)>=3) and (label not in unlabeled_files)):
                  audio['file_path'].append(file)
                  audio['label'].append(label)
    
        dataframe_name = pd.DataFrame(audio)
        return dataframe_name
    
    
    def split_for_training_testing(self, final_data):
        # Splitting the Data into Training Data and Testing Data in the proportion of 80:20 (Train:Test)
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_idx, test_idx in split.split(final_data, final_data.label):
            train_data = final_data.loc[train_idx]
            test_data = final_data.loc[test_idx]
            return [train_data, test_data]
        
        
    #Setting the sampling audio rate to 22050, setting the duration only to 3 seconds and finally
    # extraction of MFCC features
    def feature_extraction(self, file_path):
       y, sr = lib.load(file_path, duration=3)
       mfcc = lib.feature.mfcc(y=y, sr=sr,n_mfcc=128)
       return mfcc
        
        
        
    def create_xtrain_xtest(self, train_data, test_data):
       x_train = np.asarray([self.feature_extraction(train_data.file_path.iloc[i]) for i in (range(len(train_data)))])
       x_test = np.asarray([self.feature_extraction(test_data.file_path.iloc[i]) for i in (range(len(test_data)))])
       return [x_train, x_test]
   
   
   
    def reshape_xtrain_x_test(self, x_train, x_test):
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
        return [x_train, x_test]
    
    
    def encode_label(self, train_data, test_data):
        # Encode the labels into numbers from string values
        y_train = self.encode.fit_transform(train_data.label)
        y_test = self.encode.fit_transform(test_data.label)
        return [y_train, y_test]
    
    
    def y_train_y_test_to_categorical(self, y_train, y_test):
        # Setting 5 labels for each audio example with their probabilities
        y_train = to_categorical(y_train, num_classes=5)
        y_test = to_categorical(y_test, num_classes=5)
        return [y_train, y_test]
    
    
    #Creating a CNN model
    def CNN_model(self, n_width,n_height,n_channels,n_dropout,n_classes):
        cnn_model = Sequential()
    
        cnn_model.add(Conv2D(filters=24, kernel_size=(5,5), strides=(1, 1),
                             input_shape=(n_width,n_height,n_channels), activation ='relu'))
        cnn_model.add(MaxPooling2D((4, 2), strides=(4, 2)))
    
        cnn_model.add(Conv2D(filters=48, kernel_size=(5,5), padding = 'valid', activation ='relu'))
        cnn_model.add(MaxPooling2D((4, 2), strides=(4, 2)))
    
        cnn_model.add(Conv2D(filters=48, kernel_size=(5,5), padding = 'valid', activation ='relu'))
    
        cnn_model.add(Flatten())
        cnn_model.add(Dropout(rate=n_dropout))
    
        cnn_model.add(Dense(64, activation ='relu'))
        cnn_model.add(Dropout(rate=n_dropout))
    
        cnn_model.add(Dense(n_classes, activation ='softmax'))
    
        return cnn_model
    
    
    def cnn_model(self, x_train):
        return self.CNN_model(x_train.shape[1], x_train.shape[2], x_train.shape[3], 0.5, len(self.encode.classes_))
    
    def setting_learning_rate_loss_ftn(self, cnn_model):
        #Setting the learning rate and loss function for the model
        optimizer = Adam(learning_rate=0.0001)
        cnn_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)
        cnn_model.summary()
        
        
    def train_model(self, cnn_model, y_train, y_test, x_train, x_test):
        os.times
        os.chdir(self.parent_data_path)
        # Training the model over 300 times and having a batch size of 128 and saving the best model in a .hdf5 file
        epochs = 300
        batch_size = 128
        file = 'cnn_heartbeat_classifier.hdf5'
        path = os.path.join(file)
        
        file_last = 'LAST_MODEL.hdf5'
        path1 = os.path.join(file_last)
        
        checkpoints_0 = ModelCheckpoint(filepath=path, save_best_only=True, verbose=1)
        checkpoints_1 = ModelCheckpoint(filepath=path1, save_best_only=False, verbose=1)
        
        cnn_history = cnn_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test),
                                    callbacks=[checkpoints_0, checkpoints_1], verbose=1)
        return cnn_history
    
    
    def display_best_training_accruracy_loss_score(self, cnn_model, x_train, y_train):
        # Displaying the best training accuracy and loss score
        training_scores = cnn_model.evaluate(x_train, y_train)
        return training_scores
    
    
    def display_best_testing_accruracy_loss_score(self, cnn_model, x_test, y_test):
        # Displaying the best training accuracy and loss score
        testing_scores = cnn_model.evaluate(x_test, y_test)
        return testing_scores
    
    
    def calculate_display_precision(self, cnn_model, x_test, y_test):
        #Calculating and displaying the Precison, Recall and F1 score for each class
        preds = cnn_model.predict(x_test)
        y_actual = []
        y_pred = []
        
        labels = self.encode.classes_
        for idx, pred in enumerate(preds):
            y_actual.append(labels[np.argmax(y_test[idx])])
            y_pred.append(labels[np.argmax(pred)])
        
        print(classification_report(y_pred, y_actual))
        return [y_actual, y_pred]
        
        
    def get_cnn_history(self, cnn_history):
        return cnn_history.history
    
    
    def plot_epoch_vs_train_test_accuracy_graph(self, cnn_history_history):
        #Plotting epoch vs Training and Testing accuracy Graph
        plt.figure(figsize=(16,6))
        plt.plot(cnn_history_history['accuracy'], color = "b")
        plt.plot(cnn_history_history['val_accuracy'], color = 'r')
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Training Accuracy','Testing Accuracy'],loc='upper left')
        
    
    def plot_epoch_vs_train_test_loss_graph(self, cnn_history_history):
        #Plotting epoch vs Training and Testing loss Graph
        plt.figure(figsize=(16,8))
        plt.plot(cnn_history_history['loss'], color = "b")
        plt.plot(cnn_history_history['val_loss'], color="r")
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Training Loss','Testing Loss'],loc='upper right')
    
    
    def confusion_matrix(self, y_actual,y_pred):
        #Creating a confusion matrix
        mat = confusion_matrix(y_actual,y_pred)
        confusion_matrix(conf_mat=mat ,figsize=(10,10),show_normed=True)
        plt.show()
        
        
    def get_restored_model_accuracy(self, cnn_model_value, x_test, y_test):
        loss, acc = cnn_model_value.evaluate(x_test, y_test, verbose=2)
        return 'Restored model, accuracy: {:5.2f}%'.format(100 * acc)
    
    
    def save_model(self, cnn_model_value, file_name): #save without extension or my_model.h5
        cnn_model_value.save('file_name')
        
        
    def get_model_from_file(self, file_path): #return in tensorflow format
        fetched_model = tf.keras.models.load_model(file_path)
        return fetched_model
    
    def convert_model_to_tflite(self, file_name):
        keras_model = tf.keras.models.load_model(file_name) #my_model.h5
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.experimental_new_converter = True
        tflite_model = converter.convert()
        open(file_name+".tflite", "wb").write(tflite_model)
        
        
        
    def run_training_model(self):
        #Creates object of the TrainHeartBeatData class type to use to access class variables
        trainHeartBeatData = TrainHeartBeatData()
        
        #Creating a dataframe for the input audio data
        raw_data = self.create_dataframe("raw_data")
        raw_data
        
        
        #move to desired working directory, preferably where your datasets sits
        os.chdir(self.data_path)
        
        #creates working directory
        os.mkdir('working')
        
        #change to newly created directory
        os.chdir('working')
        
        #create Data2 folder
        os.mkdir('Data2')
        
        #copy files in set_a and set_b files into working/Data2 folder
        trainHeartBeatData.fill_folder1_toData2("set_a")
        trainHeartBeatData.fill_folder1_toData2("set_b")
        
        #confirm you are in the 'working' directory and create the OUT folder inside it
        os.getcwd()
        #create OUT folder
        os.mkdir('OUT')
        
        #Checking and creating new directory for saving newly generated audio files using data augmentation
        trainHeartBeatData.create_new_augmented_data_files()
        
        
        #Creating new dataframe from the Updated Audio Dataset
        final_data = trainHeartBeatData.create_dataframe_final("final_data")
        final_data
        
        #Counting the number of lables in the final dataframe
        final_data.label.value_counts()
        
        #Data Distribution Graph
        trainHeartBeatData.data_distribution(final_data)
        
        #Splitting the Data into Training Data and Testing Data in the proportion of 80:20 (Train:Test)
        split_train_test_data = trainHeartBeatData.split_for_training_testing(final_data)
        #training data
        train_data = split_train_test_data[0]
        test_data = split_train_test_data[1]
        #display total training data
        train_data
        #display total testing data
        test_data
        
        #Create x_train and x_test for training and testing
        x_train_x_test = trainHeartBeatData.create_xtrain_xtest(train_data, test_data)
         #x_train data
        x_train = x_train_x_test[0]
        #x_test data
        x_test = x_train_x_test[1]
        
        #display all x_train data
        x_train
        #display all x_test data
        x_test
        
        
        #Changing the shape of the Training and Testing inputs to (3340, 128, 130, 1) to meet the CNN input requirements
        reshape_x_train_x_test = trainHeartBeatData.reshape_xtrain_x_test(x_train, x_test)
        x_train = reshape_x_train_x_test[0]
        x_test = reshape_x_train_x_test[1]
        #Print x_train and x_test reshape tuple
        print("X_Train Shape: ", x_train.shape)
        print("X_Test Shape: ", x_test.shape)
        
        
        #Encode the labels into numbers from string values
        encode_label_value = trainHeartBeatData.encode_label(train_data, test_data)
        
        #Get y_train and y_test
        y_train = encode_label_value[0]
        y_test = encode_label_value[1]

        #Setting 5 labels for each audio example with their probabilities
        y_train_y_test_value = trainHeartBeatData.y_train_y_test_to_categorical(y_train, y_test)
        
        #update y_train and y_test data
        y_train = y_train_y_test_value[0]
        y_test = y_train_y_test_value[1]
        
        #Print new shape
        print("Y_Train Shape: ", y_train.shape)
        print("Y_Test Shape: ", y_test.shape)
        
        #Create CNN model
        cnn_model_value = trainHeartBeatData.cnn_model(x_train)
        
        #Gap between logs
        print('||||||||||||||||||||||||||||||||||||||||||||||||||||||' + '\n')
        
        #Setting the learninig rate and Loss function for the model
        trainHeartBeatData.setting_learning_rate_loss_ftn(cnn_model_value)
        
        print('||||||||||||||||||||||||||||||||||||||||||||||||||||||' + '\n')
        
        #set start time for CPU Processing time
        st = time.process_time()
        
        #CNN history
        cnn_history = trainHeartBeatData.train_model(cnn_model_value, y_train, y_test, x_train, x_test)
        
        #set end time for CPU Processing time
        et = time.process_time()
        
        #process time CPU time difference
        res = et - st
        
        #Gap
        print('||||||||||||||||||||||||||||||||||||||||||||||||||||||' + '\n')
        #Print CPU time execution
        print('CPU Execution time:', res, 'seconds'+'\n')


        #Displaying the best training accuracy and loss score
        training_scores = trainHeartBeatData.display_best_training_accruracy_loss_score(cnn_model_value, x_train, y_train)
        #Print scores
        print('Least Training Loss:', training_scores[0])
        print('Best Training Accuracy:', training_scores[1])
        
        #Gap
        print('||||||||||||||||||||||||||||||||||||||||||||||||||||||' + '\n')
        
        #Displaying the best testing accuracy and loss score
        testing_scores = trainHeartBeatData.display_best_testing_accruracy_loss_score(cnn_model_value, x_test, y_test)
        #Print scores
        print('Least Testing Loss:', testing_scores[0])
        print('Best Testing Accuracy:', testing_scores[1])


        #Gap
        print('||||||||||||||||||||||||||||||||||||||||||||||||||||||' + '\n')
        
        #
        y_actual_y_pred = trainHeartBeatData.calculate_display_precision(cnn_model_value, cnn_model_value, x_test, y_test)
        y_actual = y_actual_y_pred[0]
        y_pred = y_actual_y_pred[1]
        
        #
        y_actual
        y_pred
        
        
        #Gap
        print('||||||||||||||||||||||||||||||||||||||||||||||||||||||' + '\n')

        history = trainHeartBeatData.get_cnn_history(cnn_history)
        
        #Gap
        print('||||||||||||||||||||||||||||||||||||||||||||||||||||||' + '\n')
        
        #display history cnn_model
        history
        
        
        #Gap
        print('||||||||||||||||||||||||||||||||||||||||||||||||||||||' + '\n')

        #        
        trainHeartBeatData.plot_epoch_vs_train_test_accuracy_graph(history)
        
        #Gap
        print('||||||||||||||||||||||||||||||||||||||||||||||||||||||' + '\n')
        
        #
        trainHeartBeatData.plot_epoch_vs_train_test_loss_graph(history)
        
        #
        os.mkdir('saved_model')
        
        #Save model without extension, ensure you are in the directory you want to save file
        trainHeartBeatData.save_model(cnn_model_value, 'saved_model/my_model') #e.g. my_model.h5
        
        #change in directory where file is
        new_model = trainHeartBeatData.get_model_from_file('my_model')
        
        #just to check the architecture of your model
        new_model.summary() 

        
        #Gap
        print('||||||||||||||||||||||||||||||||||||||||||||||||||||||' + '\n')
        
        #
        restored_model_accuracy = trainHeartBeatData.get_restored_model_accuracy(cnn_model_value, x_test, y_test)
        print(restored_model_accuracy + '\n')
        
        #convert training to tflite
        trainHeartBeatData.convert_model_to_tflite('heartbeat_classifier_model')




#run model, you can modify 
t = TrainHeartBeatData()
t.run_training_model()











        
        
        

                      
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 
                 