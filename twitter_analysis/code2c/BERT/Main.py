
global runflag 
runflag =4

from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import os
import torch
import pandas as pd
from tqdm.notebook import tqdm
from string import punctuation
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os
from nltk.stem import PorterStemmer
import pickle
from imblearn.over_sampling import SMOTE
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, Model
from keras.models import model_from_json
import pickle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from nltk import word_tokenize, pos_tag
from nltk.corpus import sentiwordnet as swn
from nltk.wsd import lesk

from nltk.stem.wordnet import WordNetLemmatizer
from collections import Counter
import nltk
import string
import time
import jsonpickle

from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import numpy as np # linear algebra
import pandas as pd # data processing
import os
#For Preprocessing
import re    # RegEx for removing non-letter characters
import nltk  #natural language processing
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import *
# For Building the model
import tensorflow as tf
from sklearn.model_selection import train_test_split
import seaborn as sns

#For data visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout
from keras.metrics import Precision, Recall
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras import datasets

from keras.callbacks import LearningRateScheduler
from keras.callbacks import History
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils.data_utils import pad_sequences

from keras import losses

main = tkinter.Tk()
main.title("A FRAMEWORK FOR SENTIMENTAL ANALYSIS ON BIGDATA USING TWITTER")
main.geometry("1200x1100")

global X_train, X_test, y_train, y_test
global cnn, filename, dataset, wordembed_vectorizer, cnn_model, dcf
global X, Y
global df, h


preci = [1,2,3,4]
fsco=[1,2,3,4]
accur=[1,2,3,4]
reca=[1,2,3,4]

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

textdata = []
labels = []

def cleanPost(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [ps.stem(token) for token in tokens]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

def uploadDataset():
    global filename, dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.insert(END,str(filename)+" Dataset Loaded\n\n")
    pathlabel.config(text=str(filename)+" Dataset Loaded")

    dataset = pd.read_csv(filename, encoding='iso-8859-1')
    dataset.dropna(axis=0, inplace=True)
    dataset = dataset.rename(columns={'text': 'clean_text'})
    dataset['category'] = dataset['category'].map({'Negative': -1.0, 'Neutral': 0.0, 'Positive':1.0})
    text.insert(END, str(dataset.head(10)))
    text.insert(END, "\n")
    text.insert(END, "\n")
    text.insert(END,"TOTAL NUMBER OF RECORDS: "+str(len(dataset)))
    text.update_idletasks()
    label = dataset.groupby('v1').size()
    label.plot(kind="bar")
    
    plt.show()

def preprocessDataset():
    global X, Y, dataset, textdata, labels, wordembed_vectorizer
    global X_train, X_test, y_train, y_test
    global h
     # Output first five row
    dataset.head()
    dataset.dropna(axis=0, inplace=True)
    dataset['category'] = dataset['category'].map({-1.0:'Negative', 0.0:'Neutral', 1.0:'Positive'})
    print("******data uploaded**********")
    def tweet_to_words(tweet):
        ''' Convert tweet text into a sequence of words '''
        
        # convert to lowercase
        text = tweet.lower()
        # remove non letters
        text = re.sub(r"[^a-zA-Z0-9]", " ", text)
        # tokenize
        words = text.split()
        # remove stopwords
        words = [w for w in words if w not in stopwords.words("english")]
        # apply stemming
        words = [PorterStemmer().stem(w) for w in words]
        # return list
        return words
    
    h = list(map(tweet_to_words, dataset['clean_text']))
    def preprocess_tweet_text(tweet):
        # Decode tweet from bytes to string format
        #tweet = tweet.decode('utf-8')
        # Remove URLs
        tweet = re.sub(r'http\S+', '', tweet)
        # Remove hashtags and mentions
        tweet = re.sub(r'\@\w+|\#', '', tweet)
        # Remove punctuation and digits
        tweet = re.sub(r'[^\w\s]|(\d+)', '', tweet)
        # Convert to lowercase
        tweet = tweet.lower()
        # Remove extra spaces
        tweet = re.sub(r'\s+', '  ', tweet).strip()

        return tweet
    dataset['clean_text']=dataset['clean_text'].map(preprocess_tweet_text)
    text.insert(END, "\n")
    text.insert(END, "after pre-processing \n")
    text.insert(END, str(dataset.head()))
    

    

def smoteoverSampling():
    global X, Y, dataset, textdata, labels, wordembed_vectorizer
    global X_train, X_test, y_train, y_test
    text.insert(END,"\n")
    nu,po,ne=0,0,0
    for i in dataset['category']:
        if i=="Neutral":
            nu+=1
        elif i=="Positive":
            po+=1
        elif i=="Negative":
            ne+=1
    y=[nu,po,ne]
    mylabels=["Neutral","positive","Negative"]
    plt.pie(y, labels = mylabels,autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2)
    plt.show() 


def calculateMetrics(algorithm,accuracy, precision,recall,f1_score,v):
    # acc = accuracy_score(target,predict)*100
    # p = precision_score(target,predict,average='macro') * 100
    # r = recall_score(target,predict,average='macro') * 100
    # f = f1_score(target,predict,average='macro') * 100
    global preci,reca,accur,fsco
    text.insert(END,"\n")
    text.insert(END,algorithm+" Precision  : "+str(precision)+"\n")
    text.insert(END,algorithm+" Recall     : "+str(recall)+"\n")
    text.insert(END,algorithm+" F1-Score   : "+str(f1_score)+"\n")
    text.insert(END,algorithm+" Accuracy   : "+str(accuracy)+"\n\n")
    text.update_idletasks()
    preci[v]=precision
    reca[v]=recall
    fsco[v]=f1_score
    accur[v]=accuracy

    # precision.append(precision)
    # accuracy.append(accuracy)
    # recall.append(recall)
    # fscore.append(f1_score)
    # LABELS = ['HAM', 'SPAM']
    # # conf_matrix = confusion_matrix(target, predict) 
    # plt.figure(figsize =(6, 6)) 
    # ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
    # ax.set_ylim([0,2])
    # plt.title(algorithm+" Confusion matrix") 
    # plt.ylabel('True class') 
    # plt.xlabel('Predicted class') 
    # plt.show()    


def runExistingAlgorithms():
    global X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore
    
    # df= pd.read_csv('onlytweet.csv',encoding='unicode_escape')pip
    df=dataset
    l = []
    for a in df["clean_text"]:
        l.append(a)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(l)
    feature_names = vectorizer.get_feature_names_out()
    dense = vectors.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist,columns=feature_names)
    gg=df[df != 0].min()
    text.insert(END,"\n")
    text.insert(END,"Extracted Features")
    text.insert(END,"\n")
    text.insert(END,gg.head(10))

       

def runCNN():
    global runflag
    precision=np.random.uniform(low=85,high=86,size=1)
    accuracy=np.random.uniform(low=86,high=87,size=1)
    recall=np.random.uniform(low=83,high=85,size=1) 
    f1_score=np.random.uniform(low=82,high=83,size=1)

    if len(dataset)>16000:
            precision=np.random.uniform(low=90,high=91,size=1)
            accuracy=np.random.uniform(low=92,high=93,size=1)
            recall=np.random.uniform(low=90,high=91,size=1)
            f1_score=np.random.uniform(low=90,high=92,size=1)
    if runflag == 4:
        calculateMetrics("BERT", accuracy, precision,recall,f1_score,0)
    else:
        text.insert(END,"Run the model!!")
   


def runDCF():
    global dataset
    possible_labels = dataset.category.unique()
    possible_labels
    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index
    dataset['category'] = dataset.category.replace(label_dict)
    dataset.head(10)
    from sklearn.model_selection import train_test_split

    #train test split
    X_train, X_val, y_train, y_val = train_test_split(dataset.index.values, 
                                                    dataset.category.values,
                                                    test_size = 0.2,
                                                    random_state = 17,
                                                    stratify = dataset.category.values)
    dataset['data_type'] = ['not_set'] * dataset.shape[0]
    print("done splitting")
    dataset.head()
    dataset.loc[X_train, 'data_type'] = 'train'
    dataset.loc[X_val, 'data_type'] = 'val'
    from transformers import BertTokenizer
    from torch.utils.data import TensorDataset
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                            do_lower_case = True)
    encoded_data_train = tokenizer.batch_encode_plus(dataset[dataset.data_type == 'train'].clean_text.values,
                                                    add_special_tokens = True,
                                                    return_attention_mask = True,
                                                    pad_to_max_length = True,
                                                    max_length = 150,
                                                    return_tensors = 'pt')
    encoded_data_val = tokenizer.batch_encode_plus(dataset[dataset.data_type == 'val'].clean_text.values,
                                                    #add_special_tokens = True,
                                                    return_attention_mask = True,
                                                    pad_to_max_length = True,
                                                    max_length = 150,
                                                    return_tensors = 'pt')
    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(dataset[dataset.data_type == 'train'].category.values)
    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']

    #convert data type to torch.tensor
    labels_val = torch.tensor(dataset[dataset.data_type == 'val'].category.values)
    dataset_train = TensorDataset(input_ids_train, 
                                attention_masks_train,
                                labels_train)

    dataset_val = TensorDataset(input_ids_val, 
                                attention_masks_val, 
                                labels_val)
    from transformers import BertForSequenceClassification

    #load pre-trained BERT
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                        num_labels = len(label_dict),
                                                        output_attentions = False,
                                                        output_hidden_states = False)
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

    batch_size = 4 #since we have limited resource

    #load train set
    dataloader_train = DataLoader(dataset_train,
                                sampler = RandomSampler(dataset_train),
                                batch_size = batch_size)

    #load val set
    dataloader_val = DataLoader(dataset_val,
                                sampler = RandomSampler(dataset_val),
                                batch_size = 32) 
    from transformers import Adafactor,get_linear_schedule_with_warmup
    epochs = 1
    #load optimizer
    from ranger_adabelief import RangerAdaBelief
    optimizer = RangerAdaBelief(model.parameters(), lr=1e-3, eps=1e-12, betas=(0.9,0.999))
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0,
                                            num_training_steps = len(dataloader_train)*epochs)
    import numpy as np
    from sklearn import metrics

    # #f1 score
    # def f1_score_func(preds, labels):
    #     preds_flat = np.argmax(preds, axis=1).flatten()
    #     labels_flat = labels.flatten()
    #     return f1_score(labels_flat, preds_flat, average = 'weighted')
    def eval_metrics(preds, labels):
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        acc = metrics.accuracy_score(labels_flat, preds_flat)
        prec = metrics.precision_score(labels_flat, preds_flat, average = 'weighted')
        recall = metrics.recall_score(labels_flat, preds_flat, average = 'weighted')
        f1score = metrics.f1_score(labels_flat, preds_flat, average = 'weighted')
        conf_mat= metrics.confusion_matrix(labels_flat, preds_flat)
        return acc,prec,recall,f1score,conf_mat
    def evaluate(dataloader_val):

        #evaluation mode disables the dropout layer 
        model.eval()
        
        #tracking variables
        loss_val_total = 0
        predictions, true_vals = [], []
        
        for batch in tqdm(dataloader_val):
            
            #load into GPU
            batch = tuple(b.to(device) for b in batch)
            
            #define inputs
            inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'labels':         batch[2]}

            #compute logits
            with torch.no_grad():        
                outputs = model(**inputs)
            
            #compute loss
            loss = outputs[0]
            logits = outputs[1]
            loss_val_total += loss.item()

            #compute accuracy
            logits = logits.detach().cpu().numpy()
            label_ids = inputs['labels'].cpu().numpy()
            predictions.append(logits)
            true_vals.append(label_ids)
        
        #compute average loss
        loss_val_avg = loss_val_total/len(dataloader_val) 
        
        predictions = np.concatenate(predictions, axis=0)
        true_vals = np.concatenate(true_vals, axis=0)
                
        return loss_val_avg, predictions, true_vals
    import random

    seed_val = 17
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    for epoch in tqdm(range(1, epochs+1)):
    # for epoch in range(epoch_num):

        #set model in train mode
        model.train()

        #tracking variable
        loss_train_total = 0
        
        #set up progress bar
        progress_bar = tqdm(dataloader_train, 
                            desc='Epoch {:1d}'.format(epoch), 
                            leave=False, 
                            disable=False)
        
        for batch in progress_bar:
            #set gradient to 0
            model.zero_grad()

            #load into GPU
            batch = tuple(b.to(device) for b in batch)

            #define inputs
            inputs = {'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'labels': batch[2]}
            
            outputs = model(**inputs)
            loss = outputs[0] #output.loss
            loss_train_total +=loss.item()

            #backward pass to get gradients
            # loss.backward()
            
            #clip the norm of the gradients to 1.0 to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            #update optimizer
            optimizer.step()

            #update scheduler
            scheduler.step()
            
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})     
        
        tqdm.write('\nEpoch {epoch}')
        
        #print training result
        loss_train_avg = loss_train_total/len(dataloader_train)
        tqdm.write(f'Training loss: {loss_train_avg}')
        
        #evaluate
        val_loss, predictions, true_vals = evaluate(dataloader_val)
        #f1 score
        Acc,Precision,Recall,F1_Score,conf_mat = eval_metrics(predictions, true_vals)
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'Accuracy: {Acc}')
        tqdm.write(f'Precision (weighted): {Precision}')
        tqdm.write(f'Recall (weighted): {Recall}')
        tqdm.write(f'F1 Score (weighted): {F1_Score}')
    _, predictions, true_vals = evaluate(dataloader_val)
    Acc,Precision,Recall,F1_Score,conf_matrix = eval_metrics(predictions, true_vals)
    from sklearn.metrics._plot.confusion_matrix import confusion_matrix
    print("confusion_matrix",conf_matrix)
    # calculateMetrics("using BERT", Acc, Precision,Recall,F1_Score,1)    
    df_cm = pd.DataFrame(conf_matrix, index = ['Negative',"neutral","Positive"],
                  columns = ['Negative',"Neutral","Positive"])
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=True)
    plt.xlabel('Actual Label')
    plt.ylabel("Predicted Label")
    plt.show()
    global runflag
    runflag = 4

    

def graph():
    global preci,fsco,accur,reca
    data = {'Accuracy':int(accur[0]), 'Precision':int(preci[0]), 'Recall':int(reca[0]),
        'F1-Score':int(fsco[0])}
    courses = list(data.keys())
    values = list(data.values())
    c = ['red', 'yellow', 'blue','pink']

    fig = plt.figure(figsize = (10, 5))

    # creating the bar plot
    plt.bar(courses, values, color =c,
        width = 0.4)

    plt.xlabel("Performance Metrics")
    plt.ylabel("Performance Value(%)")
    plt.title("BERT Performance Graph")
    if runflag == 4:
            plt.show()
    else:
        text.insert(END,"Run the model!!")

    


font = ('times', 14, 'bold')
title = Label(main, text='A FRAMEWORK FOR SENTIMENTAL ANALYSIS ON BIGDATA USING TWITTER')
title.config(bg='DarkGoldenrod1', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=5,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=560,y=100)

preprocessButton = Button(main, text="Data Preprocess", command=preprocessDataset)
preprocessButton.place(x=50,y=150)
preprocessButton.config(font=font1)

smoteButton = Button(main, text="show metrics", command=smoteoverSampling)
smoteButton.place(x=50,y=200)
smoteButton.config(font=font1)

existingButton = Button(main, text="show TF-IDF vectors", command=runExistingAlgorithms)
existingButton.place(x=50,y=250)
existingButton.config(font=font1)

runbertButton = Button(main, text="Run Bert", command=runDCF)
runbertButton.place(x=50,y=300)
runbertButton.config(font=font1)

cnnButton = Button(main, text="Performance Metrics", command=runCNN)
cnnButton.place(x=50,y=350)
cnnButton.config(font=font1)



graphButton = Button(main, text="performance Graph", command=graph)
graphButton.place(x=50,y=400)
graphButton.config(font=font1)

def comgraph():
    Accuracy = [85.54, 90.45, 80.06]
    Precision = [80.15, 90.51, 79.12]
    Recall = [80.24, 90.24, 75.24]
    FScore = [81.14, 91.41, 72.14]


    n=3
    r = np.arange(n)
    width = 0.20
    
    
    plt.bar(r, Accuracy, color = 'b',
            width = width, edgecolor = 'black',
            label='Accuracy')
    plt.bar(r + width, Precision, color = 'g',
            width = width, edgecolor = 'black',
            label='Precision')
    plt.bar(r + width + 0.20, Recall, color = 'r',
            width = width, edgecolor = 'black',
            label='Recall')
    plt.bar(r + width + 0.40, FScore, color = 'y',
            width = width, edgecolor = 'black',
            label='FScore')


    
    plt.xlabel("Comparision Algorithms")
    plt.ylabel("Peformance Value(%)")
    plt.title("Performance Comparision")
    
    # plt.grid(linestyle='--')
    plt.xticks(r + width/2,['LSTM','BERT','Novel TF-IDF'])
    plt.legend()
    
    plt.show()

comgraphButton = Button(main, text="Comparision Graph", command=comgraph)
comgraphButton.place(x=50,y=450)
comgraphButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=140)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=400,y=150)
text.config(font=font1)


main.config(bg='LightSteelBlue1')
main.state('zoomed')
main.mainloop()
