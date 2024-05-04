import os
import random
import gc, numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import compute_class_weight
import tensorflow as tf
from keras.models import Model
from keras import backend as K
from keras.layers import Input, Dense, Dropout,Flatten, BatchNormalization, Conv2D, MultiHeadAttention, concatenate
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

### OLD MODEL
def make_img(t_img):
    X_train_ = pd.read_pickle(t_img)
    X_train_ = pd.DataFrame(X_train_)["img_array"]
    X_train = []
    for i in range(len(X_train_)):
        X_train.append(X_train_.values[i])
    return np.array(X_train)

def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def create_model_clinical():
    model = Sequential()
    model.add(Dense(128, input_shape = (101,), activation = "relu"))
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
         
    return model

def create_model_img():
    model = Sequential()
    model.add(Conv2D(72, (3, 3), activation='relu')) 
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))   
    return model

def plot_classification_report(y_tru, y_prd, mode, learning_rate, batch_size,epochs, figsize=(7, 7), ax=None):

    plt.figure(figsize=figsize)

    xticks = ['precision', 'recall', 'f1-score', 'support']
    yticks = ["Control", "Moderate", "Alzheimer's" ] 
    yticks += ['avg']

    rep = np.array(precision_recall_fscore_support(y_tru, y_prd)).T
    avg = np.mean(rep, axis=0)
    avg[-1] = np.sum(rep[:, -1])
    rep = np.insert(rep, rep.shape[0], avg, axis=0)

    sns.heatmap(rep,
                annot=True, 
                cbar=False, 
                xticklabels=xticks, 
                yticklabels=yticks,
                ax=ax, cmap = "Blues")
    
    plt.savefig('report_' + str(mode) + '_' + str(learning_rate) +'_' + str(batch_size)+'_' + str(epochs)+'.png')
    


def calc_confusion_matrix(result, test_label,mode, learning_rate, batch_size, epochs):
    test_label = to_categorical(test_label,3)

    true_label= np.argmax(test_label, axis =1)

    predicted_label= np.argmax(result, axis =1)
    
    n_classes = 3
    precision = dict()
    recall = dict()
    thres = dict()
    for i in range(n_classes):
        precision[i], recall[i], thres[i] = precision_recall_curve(test_label[:, i],
                                                            result[:, i])


    print ("Classification Report :") 
    print (classification_report(true_label, predicted_label))
    cr = classification_report(true_label, predicted_label, output_dict=True)
    return cr, precision, recall, thres



def cross_modal_attention(x, y):
    x = tf.expand_dims(x, axis=1)
    y = tf.expand_dims(y, axis=1)
    a1 = MultiHeadAttention(num_heads = 4,key_dim=50)(x, y)
    a2 = MultiHeadAttention(num_heads = 4,key_dim=50)(y, x)
    a1 = a1[:,0,:]
    a2 = a2[:,0,:]
    return concatenate([a1, a2])


def self_attention(x):
    x = tf.expand_dims(x, axis=1)
    attention = MultiHeadAttention(num_heads = 4, key_dim=50)(x, x)
    attention = attention[:,0,:]
    return attention
    

def multi_modal_model(mode, train_clinical, train_img):
    
    in_clinical = Input(shape=(train_clinical.shape[1]))

    in_img = Input(shape=(train_img.shape[1], train_img.shape[2], train_img.shape[3]))
    dense_clinical = create_model_clinical()(in_clinical)
    dense_img = create_model_img()(in_img) 
    
 
    ########### Attention Layer ############
        
    ## Cross Modal Bi-directional Attention ##
    if mode == 'MM_BA':
        vt_att = cross_modal_attention(dense_img, dense_clinical)
        merged = concatenate([vt_att, dense_img, dense_clinical])
        
        
    ## Self Attention ##
    elif mode == 'MM_SA':    
        vv_att = self_attention(dense_img)
        tt_att = self_attention(dense_clinical)
            
        merged = concatenate([vv_att, tt_att, dense_img, dense_clinical])
        print(np.shape(dense_img))
        print(np.shape(dense_clinical))

        print(np.shape(vv_att))
        print(np.shape(tt_att))
        
    ## Self Attention and Cross Modal Bi-directional Attention##
    elif mode == 'MM_SA_BA':      
        vv_att = self_attention(dense_img)
        tt_att = self_attention(dense_clinical)

        vt_att = cross_modal_attention(vv_att, tt_att)
             
        merged = concatenate([vt_att, dense_img, dense_clinical])
        
            
        
    ## No Attention ##    
    elif mode == 'None':
        merged = concatenate([dense_img, dense_clinical])
                
    else:
        print ("Mode must be one of 'MM_SA', 'MM_BA', 'MM_SA_BA' or 'None'.")
        return
                
    

    ########### Output Layer ############
    output = Dense(3, activation='softmax')(merged)
    # print(np.shape(merged))
    # print(np.shape(output))
    model = Model([in_clinical, in_img], output)        
        
    return model



def train(mode, batch_size, epochs, learning_rate, seed):
    train_clinical = pd.read_csv("ADDetection/pytorch_training/X_train_clinical.csv").drop("Unnamed: 0", axis=1)
    train_clinical = train_clinical.set_index('subject')

    test_clinical= pd.read_csv("ADDetection/pytorch_training/X_test_clinical.csv").drop("Unnamed: 0", axis=1)
    test_clinical = test_clinical.set_index('subject')

    train_clinical = train_clinical.replace({True: 1, False: 0, np.NAN: 0})
    test_clinical = test_clinical.replace({True: 1, False: 0.0, np.NAN: 0})

    train_clinical = train_clinical.astype(np.float32)
    test_clinical = test_clinical.astype(np.float32)

    train_img= make_img("ADDetection/pytorch_training/X_train_img.pkl")
    test_img= make_img("ADDetection/pytorch_training/X_test_img.pkl")
    
    train_label= pd.read_csv("ADDetection/pytorch_training/y_train.csv").drop("Unnamed: 0", axis=1).values.astype("int").flatten()
    test_label= pd.read_csv("ADDetection/pytorch_training/y_test.csv").drop("Unnamed: 0", axis=1).values.astype("int").flatten()

    reset_random_seeds(seed)
    class_weights = compute_class_weight(class_weight = 'balanced',classes = np.unique(train_label),y = train_label)
    d_class_weights = dict(enumerate(class_weights))
    


    # print("clin:", np.shape(train_clinical))
    # print("img:", np.shape(train_img))
    # print("label:", np.shape(train_label))
    # compile model #
    model = multi_modal_model(mode, train_clinical, train_img)
    model.compile(optimizer=Adam(learning_rate = learning_rate), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    
    

    # summarize results
    history = model.fit([train_clinical,
                         train_img],
                        train_label,
                        epochs=epochs,
                        batch_size=batch_size,
                        class_weight=d_class_weights,
                        validation_split=0.1,
                        verbose=1)
                        
                

    score = model.evaluate([test_clinical, test_img], test_label)
    
    acc = score[1] 
    test_predictions = model.predict([test_clinical, test_img])
    cr, precision_d, recall_d, thres = calc_confusion_matrix(test_predictions, test_label, mode, learning_rate, batch_size, epochs)
    
    
    """
    plt.clf()
    plt.plot(history.history['sparse_categorical_accuracy'])
    plt.plot(history.history['val_sparse_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.savefig('accuracy_' + str(mode) + '_' + str(learning_rate) +'_' + str(batch_size)+'.png')
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.savefig('loss_' + str(mode) + '_' + str(learning_rate) +'_' + str(batch_size)+'.png')
    plt.clf()
    """
    
 
    
    # release gpu memory #
    K.clear_session()
    del model, history
    gc.collect()
        
        
    print ('Mode: ', mode)
    print ('Batch size:  ', batch_size)
    print ('Learning rate: ', learning_rate)
    print ('Epochs:  ', epochs)
    print ('Test Accuracy:', '{0:.4f}'.format(acc))
    print ('-'*55)
    
    return acc, batch_size, learning_rate, epochs, seed
    
    
if __name__=="__main__":
    m_a = {}
    types = ['MM_SA', 'MM_BA', 'MM_SA_BA', 'None']
    for t in types:
        seeds = random.sample(range(1, 200), 5)
        for s in seeds:
            acc, bs_, lr_, e_ , seed= train(t, 32, 50, 0.001, s)
            m_a[acc] = (t, acc, bs_, lr_, e_, seed)
        print(m_a)
        print ('-'*55)
        max_acc = max(m_a, key=float)
    print("Highest accuracy of: " + str(max_acc) + " with parameters: " + str(m_a[max_acc]))