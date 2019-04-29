'''
================================================ 
##            VOICEBOOK REPOSITORY            ##      
================================================ 

repository name: voicebook 
repository version: 1.0 
repository link: https://github.com/jim-schwoebel/voicebook 
author: Jim Schwoebel 
author contact: js@neurolex.co 
description: a book and repo to get you started programming voice applications in Python - 10 chapters and 200+ scripts. 
license category: opensource 
license: Apache 2.0 license 
organization name: NeuroLex Laboratories, Inc. 
location: Seattle, WA 
website: https://neurolex.ai 
release date: 2018-09-28 

This code (voicebook) is hereby released under a Apache 2.0 license license. 

For more information, check out the license terms below. 

================================================ 
##               LICENSE TERMS                ##      
================================================ 

Copyright 2018 NeuroLex Laboratories, Inc. 

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

     http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 

================================================ 
##               SERVICE STATEMENT            ##        
================================================ 

If you are using the code written for a larger project, we are 
happy to consult with you and help you with deployment. Our team 
has >10 world experts in Kafka distributed architectures, microservices 
built on top of Node.js / Python / Docker, and applying machine learning to 
model speech and text data. 

We have helped a wide variety of enterprises - small businesses, 
researchers, enterprises, and/or independent developers. 

If you would like to work with us let us know @ js@neurolex.co. 

================================================ 
##              LOAD_AUDIOTPOT.PY             ##    
================================================ 

Fingerprint audio models in a streaming folder. 
'''

################################################
##              Import statements             ##    
################################################

import librosa, pickle, getpass, time, shutil, sys
from pydub import AudioSegment
import speech_recognition as sr  
import os, nltk, random, json 
import numpy as np 
import pandas as pd 
import librosa_features as lf 

################################################
##               Loading settings.            ##    
################################################

# these are from settings.json file (helps us with applying ML models)
g=json.load(open('settings.json'))
overlapping = g['overlapping']
plot_feature = g['plot_feature']
probability_default = g['probability_default']
probability_labeltype = g['probability_labeltype']
timesplit=g['timesplit']
visualize_feature = g['visualize_feature']

################################################
##                 Helper functions           ##    
################################################

## helper function to get wav files 
def find_wav(listdir):
    wavfiles=list()
    for j in range(len(listdir)):
        if listdir[j][-4:]=='.wav':
            wavfiles.append(listdir[j])
    return wavfiles 

# get statistical features in numpy
def stats(matrix):

    try:
        mean=np.mean(matrix)
        std=np.std(matrix)
        maxv=np.amax(matrix)
        minv=np.amin(matrix)
        median=np.median(matrix)

        output=np.array([mean,std,maxv,minv,median])
    except:
        output='error'
    
    return output

def exportfile(newAudio,time1,time2,filename,i):
    #Exports to a wav file in the current path.
    newAudio2 = newAudio[time1:time2]
    g=os.listdir()
    if filename[0:-4]+'_'+str(i)+'.wav' in g:
        filename2=str(i)+'_segment'+'.wav'
        print('making %s'%(filename2))
        newAudio2.export(filename2,format="wav")
    else:
        filename2=filename[0:-4]+'_'+str(i)+'.wav'
        print('making %s'%(filename2))
        newAudio2.export(filename2, format="wav")

    return filename2 

def split_segments(filename, timesplit):
    #recommend >0.20 seconds for timesplit 
    hop_length = 512
    n_fft=2048
    
    y, sr = librosa.load(filename)
    duration=float(librosa.core.get_duration(y))
    
    #Now splice an audio signal into individual elements of 20 ms and extract
    segnum=round(duration/timesplit)
    deltat=duration/segnum
    timesegment=list()
    time=0

    for i in range(segnum):
        #milliseconds
        timesegment.append(time)
        time=time+deltat*1000

    newAudio = AudioSegment.from_wav(filename)
    filelist=list()
    file=filename
    
    for i in range(len(timesegment)-1):
        filename=exportfile(newAudio,timesegment[i],timesegment[i+1],file,i)
        filelist.append(filename)

    return filelist 

def featurize(wavfile):
    features, labels = lf.librosa_featurize(wavfile, False)
    return features.tolist()

# insert in model name and output classes in series 
def get_classes(modelname):
    classnum=modelname.count('_')
    classes=modelname.split('_')[0:classnum]
    return classes, classnum

def model_file(features, model_dir, modelnames, wavfile):

    # make a list of classes 
    class_nums=list()
    class_list=list() 
    class_accuracies=list()

    # load the machine learniing model of interest + loop through 
    for y in range(len(modelnames)):
        # make sure you are in the proper directory 
        os.chdir(model_dir)
        
        # load the model of interest 
        modelname=modelnames[y]
        classes, classnum = get_classes(modelname)
        loadmodel=open(modelname, 'rb')
        model = pickle.load(loadmodel)
        classaccuracy=json.load(open(modelname[0:-7]+'.json'))['accuracy']

        # change to load directory to featurize wav file + model it 
        os.chdir(load_dir)
        output=str(model.predict(features)[0])

        # make this adapt to as many as N classes 
        for i in range(len(classes)):
            if float(output)==i:
                classname=classes[i]

        class_nums.append(classnum)
        class_list.append(classname)
        class_accuracies.append(classaccuracy)
        class_names.append(classes)

    return class_nums, class_list, class_accuracies, class_names

def create_csv(csvfilename, filenames, starts, stops, label_texts, probabilities):
    print(len(filenames))
    print(len(starts))
    print(len(stops))
    print(len(label_texts))
    print(len(probabilities))

    df = pd.DataFrame({'filename': filenames,
                       'onset': np.array(starts),
                       'offset': np.array(stops),
                       'event_label': np.array(label_texts),
                       'probability': np.array(probabilities)})

    print(df)

    
    df.to_csv(csvfilename)

def visualize(hostdir, wavfile, csvfile):
    os.chdir(hostdir)
    os.system('python3 ./sed_vis/visualize.py ./load_dir/%s ./load_dir/%s'%(wavfile, csvfile))

################################################
##                 Main scripts               ##    
################################################

# set directory paths 
host_dir=os.getcwd()
cur_dir=os.getcwd()+'/load_dir'
model_dir=os.getcwd()+'/models'
load_dir=os.getcwd()+'/load_dir'

# get model names 
modelnames=list()
os.chdir(model_dir)
listdir=os.listdir()
for i in range(len(listdir)):
    if listdir[i][-7:]=='.pickle':
        modelnames.append(listdir[i])

# initialize some count variables to evaluate error paths 
count=0
errorcount=0

# make a load_dir if it does not exist + go there
try:
    os.chdir(load_dir)
except:
    os.mkdir(load_dir)
    os.chdir(load_dir)
    
listdir=os.listdir()
print(os.getcwd())

# get all .WAV files in the load_dir using helper function 
wavfiles=find_wav(listdir)

# loop through all the .WAV files and apply all machine learning models in the window of interest 
for i in range(len(wavfiles)):

    os.chdir(load_dir)
    filename=wavfiles[i]

    if filename[0:-4]+'.json' not in listdir:

        foldername=filename[0:-4]
        os.mkdir(foldername)
        os.chdir(foldername)
        folder_dir=os.getcwd()

        # move file to the proper directory 
        shutil.copy(load_dir+'/'+filename, load_dir+'/'+foldername+'/'+filename)
        filelist=split_segments(filename, timesplit)
        # remove the filename from current directory 
        os.remove(filename)

        # initialize list to count audio events  
        class_nums=list()
        class_list=list()
        class_accuracies=list()
        class_names=list()

        # now iterate through timesplit length files to model each file 
        for j in range(len(filelist)):
            os.chdir(folder_dir)
            features=np.array(featurize(filelist[j]))
            print(features)
            features=features.reshape(1,-1)
            temp_class_nums, temp_class_list, temp_class_accuracies, temp_class_names =model_file(features, model_dir, modelnames, filelist[j])
            class_nums.append(temp_class_nums)
            class_list.append(temp_class_list)
            class_accuracies.append(temp_class_accuracies)
            class_names.append(temp_class_names)

        print(class_list)
        print(class_nums)
        print(class_accuracies)
        # go to directory where actual file is that is being analyzed 
        os.chdir(load_dir)

        # now count counsecutive events in timesplit window (for all events)
        master_classlist=list()

        # iterate all through class lists and count consecutive events 
        for o in range(len(class_list[0])):
            # now go through each event class list 
            temp_time=0
            class_list2=list()

            for j in range(len(class_list)):
                print('classes this time split: %s'%(class_list[j]))
                print('number of classes possible: %s'%(class_nums[j]))

                if j != 0:
                    print('analyzing... %s'%(str(o)))
                    print(class_list[j][o])
                    if class_list[j][o] == class_list[j-1][o]:
                        # this means two events happened consecutively 
                        # e.g. speech --> speech 
                        temp_time=temp_time+timesplit
                    else:
                        # this means that two events did not happen conseucutively, a shift happened 
                        # e.g. speech--> silence 
                        class_list2.append({class_list[j-1][o]:temp_time})
                        temp_time=timesplit
                else:
                    pass 

            # add in rest of frame if last frame is open and no shift happened
            if temp_time != timesplit:
                class_list2.append({class_list[j][o]:temp_time})

            # this merges all master classes 
            master_classlist.append(class_list2)

        print(master_classlist)
        event_datas=list()

        # initialize .CSV output parammeters 
        csvfilename=filename[0:-4]+'.csv'
        csvfilenames=list()
        onsets=list()
        offsets=list()
        event_labels=list()
        probabilities=list()
        

        # now make some metrics for each event that has happened in the session
        for j in range(len(master_classlist)):
            # loop through each potential class of models 
            for k in range(class_nums[0][j]):
                # initialize event list before going into main loop 
                event=class_names[j][k]
                model_name=modelnames[j]
                print('calculating %s with %s'%(event, model_name))
                event_lengths=list()
                event_lengths_array=list()

                for l in range(len(master_classlist[j])):
                    # try each length of model / class 
                    try:
                        print(master_classlist)
                        event_length=master_classlist[j][l-1][event]
                        event_lengths.append(event_length)
                        event_lengths_array.append(np.array(event_length))

                    except:
                        pass 

                    print(event_lengths)
                    print(event_lengths_array)
                    event_stats=stats(event_lengths_array)

                    # assemble proper array to count events 
                    tclasslist=list()
                    for m in range(len(class_list)):
                        tclasslist.append(class_list[m][j])

                    print(tclasslist)
                    # calculate statistical features of event lengths 
                    total_event_lengths=tclasslist.count(event)*timesplit

                    ################################################
                    ##                 Main scripts               ##    
                    ################################################

                    # now making output align with .CSV schema 

                    # filename    onset   offset  event_label probability
                    # fast.wav    0   0.2 silence 0.8

                    if k == 0:
                        # only put outputs from the first iteration in ongoing .CSV list 
                        probability=class_accuracies[j][k]
                        onset=0

                        for m in range(len(tclasslist)):
                            event_label=tclasslist[m]
                            probabilities.append(probability)
                            event_labels.append(event_label+'_prediction')
                            onsets.append(onset)
                            onset=onset+timesplit
                            offsets.append(onset)
                            csvfilenames.append(csvfilename)

                    try:
                        event_data= {'filename': filename,
                                    'total_length': total_event_lengths,
                                    'event': event,
                                    'mean':float(event_stats[0]),
                                    'std':float(event_stats[1]),
                                    'max':float(event_stats[2]),
                                    'min':float(event_stats[3]),
                                    'median':float(event_stats[4]),
                                    'model': model_name,
                                    'model accuracy': probability,
                                    'possible classes': class_names[j],
                                    'window': timesplit,
                                    }
                    except:
                        event_data= {'filename': filename,
                                    'event': event,
                                    'total_length': total_event_lengths,
                                    'mean':0,
                                    'std':0,
                                    'max':0,
                                    'min':0,
                                    'median':0,
                                    'model': model_name,
                                    'model accuracy': probability,
                                    'possible classes': class_names[j],
                                    'window': timesplit,
                                    }            

                    event_datas.append(event_data)


        # now write all this to .CSV 
        if csvfilename not in os.listdir(load_dir):
            create_csv(csvfilename, csvfilenames, onsets, offsets, event_labels, probabilities)

        os.chdir(load_dir)
        jsonfilename=filename[0:-4]+'.json'

        jsonfile=open(jsonfilename,'w')
        data={'filename': filename,
               'event_data': event_datas}
        json.dump(data,jsonfile)
        jsonfile.close()
        shutil.rmtree(foldername)

        if visualize_feature == True and sys.argv[1] != 'suppress':
            visualize(hostdir, csvfilename, filename)



 


