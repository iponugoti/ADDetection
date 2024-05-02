import pandas as pd
import numpy as np
import os
import random
import tensorflow as tf

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential


def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    #this is created in the clinical preprocess jupyter notebook
    print(os.getcwd())
    X_train = pd.read_pickle("/Users/timothypyon/Desktop/DL_Project/ADDetection/preprocess_clinical/X_train_c.pkl")
    y_train = pd.read_pickle("/Users/timothypyon/Desktop/DL_Project/ADDetection/preprocess_clinical/y_train_c.pkl")

    X_test = pd.read_pickle("/Users/timothypyon/Desktop/DL_Project/ADDetection/preprocess_clinical/X_test_c.pkl")
    y_test = pd.read_pickle("/Users/timothypyon/Desktop/DL_Project/ADDetection/preprocess_clinical/y_test_c.pkl")
    acc = []
    f1 = []
    precision = []
    recall = []
    seeds = random.sample(range(1, 200), 5)
    for seed in seeds:
        reset_random_seeds(seed)
        model = Sequential()
        model.add(Dense(128, input_shape = (100,), activation = "relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(64, activation = "relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        model.add(Dense(50, activation = "relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        
        model.add(Dense(3, activation = "softmax"))
        
        model.compile(Adam(learning_rate = 0.0001), "sparse_categorical_crossentropy", metrics = ["sparse_categorical_accuracy"])
        
        model.summary()
        
        X_train = X_train.replace({True: 1, False: 0, np.NAN: 0})
        y_train = y_train.replace({True: 1, False: 0.0, np.NAN: 0})


        history = model.fit(tf.convert_to_tensor(X_train, dtype=tf.float32), tf.convert_to_tensor(y_train, dtype=tf.float32),  epochs=100, validation_split=0.1, batch_size=32,verbose=1) 

        X_test = X_test.replace({True: 1, False: 0, np.NAN: 0})
        y_test = y_test.replace({True: 1, False: 0.0, np.NAN: 0})
        X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
        y_test_tensor = tf.convert_to_tensor(y_test, dtype=tf.float32)


        score = model.evaluate(X_test_tensor, y_test_tensor, verbose=0)
        print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
        acc.append(score[1])
        
        test_predictions = model.predict(X_test_tensor)
        test_label = to_categorical(y_test_tensor,3)

        true_label= np.argmax(test_label, axis =1)

        predicted_label= np.argmax(test_predictions, axis =1)
        
        cr = classification_report(true_label, predicted_label, output_dict=True)
        precision.append(cr["macro avg"]["precision"])
        recall.append(cr["macro avg"]["recall"])
        f1.append(cr["macro avg"]["f1-score"])
    
    print("Avg accuracy: " + str(np.array(acc).mean()))
    print("Avg precision: " + str(np.array(precision).mean()))
    print("Avg recall: " + str(np.array(recall).mean()))
    print("Avg f1: " + str(np.array(f1).mean()))
    print("Std accuracy: " + str(np.array(acc).std()))
    print("Std precision: " + str(np.array(precision).std()))
    print("Std recall: " + str(np.array(recall).std()))
    print("Std f1: " + str(np.array(f1).std()))
    print(acc)
    print(precision)
    print(recall)
    print(f1)
    
    
    """
    plt.plot(history.history['sparse_categorical_accuracy'])
    plt.plot(history.history['val_sparse_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    #plt.savefig('snp_loss.png')
    plt.show()
    """


if __name__ == '__main__':
    main()



