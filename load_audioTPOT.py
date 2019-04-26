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

import librosa, pickle, getpass, time, shutil
from pydub import AudioSegment
import speech_recognition as sr  
import os, nltk, random, json 
import numpy as np 
import librosa_features as lf 

cur_dir=os.getcwd()+'/load_dir'
model_dir=os.getcwd()+'/models'
load_dir=os.getcwd()+'/load_dir'
modelname='speech_silence_tpotclassifier.pickle'

## helper function
def find_wav(listdir):
    wavfiles=list()
    for j in range(len(listdir)):
        if listdir[j][-4:]=='.wav':
            wavfiles.append(listdir[j])
    return wavfiles 

# get statistical features in numpy
def stats(matrix):
    mean=np.mean(matrix)
    std=np.std(matrix)
    maxv=np.amax(matrix)
    minv=np.amin(matrix)
    median=np.median(matrix)

    output=np.array([mean,std,maxv,minv,median])
    
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

def split_segments(filename):
    #recommend >0.20 seconds for timesplit 
    timesplit=0.20
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

model_list=list()
os.chdir(model_dir)
listdir=os.listdir()

for i in range(len(listdir)):
    if listdir[i][-7:]=='.pickle' and listdir[i].find('tpot')>0:
        model_list.append(listdir[i])

count=0
errorcount=0

try:
    os.chdir(load_dir)
except:
    os.mkdir(load_dir)
    os.chdir(load_dir)
    
listdir=os.listdir()
print(os.getcwd())

# get all .WAV files 
wavfiles=find_wav(listdir)

# load the machine learniing model 
os.chdir(model_dir)
loadmodel=open(modelname, 'rb')
model = pickle.load(loadmodel)
i1=modelname.find('_')
name1=modelname[0:i1]
i2=modelname[i1+1:]
i3=i2.find('_')
name2=i2[0:i3]
os.chdir(load_dir)

# loop through all the .WAV files and count number of pauses per file (20 MS window)
for i in range(len(wavfiles)):

    os.chdir(load_dir)
    filename=wavfiles[i]

    if filename[0:-4]+'.json' not in listdir:

        foldername=filename[0:-4]
        os.mkdir(foldername)
        os.chdir(foldername)

        # move file to the proper directory 
        shutil.copy(load_dir+'/'+filename, load_dir+'/'+foldername+'/'+filename)
        filelist=split_segments(filename)
        # remove the filename from current directory 
        os.remove(filename)

        # initialize list to count silence events 
        class_list=list()

        for j in range(len(filelist)):
            features=np.array(featurize(filelist[j]))
            print(features)
            features=features.reshape(1,-1)
            output=str(model.predict(features)[0])

            if float(output)==0:
                classname=name1
            else:
                classname=name2

            class_list.append(classname)

        os.chdir(load_dir)

        # now count counsecutive pauses compressed into 20 millsecond windows 
        class_list2=list()
        temp_time=0
        for j in range(len(class_list)):
            if j != 0:
                if class_list[j] == class_list[j-1]:
                    # merge pause lengths and speech segments 
                    temp_time=temp_time+0.20
                else:
                    # don't merge them, indicates a shift 
                    class_list2.append({class_list[j-1]:temp_time})
                    temp_time=0.20
            else:
                pass 

        pause_lengths=list()
        pause_lengths_array=list()

        for j in range(len(class_list2)):
            try:
                pause_length=class_list2[j]['silence']
                pause_lengths.append(pause_length)
                pause_lengths_array.append(np.array(pause_length))

            except:
                pass 

        pause_stats=stats(pause_lengths_array)
        # calculate statistical features of pause lengths
        total_pause_lengths=class_list.count('silence')*0.20

        data= {'filename': filename,
               'total_length': total_pause_lengths,
               'mean':float(pause_stats[0]),
               'std':float(pause_stats[1]),
               'max_value':float(pause_stats[2]),
               'min_pause':float(pause_stats[3]),
               'median':float(pause_stats[4]),
               }

        jsonfilename=filename[0:-4]+'.json'

        jsonfile=open(jsonfilename,'w')
        json.dump(data,jsonfile)
        jsonfile.close()


