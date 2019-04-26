'''
================================================ 
          ACOUSTIC_EVENT_DETECTION REPOSITORY                     
================================================ 

repository name: acoustic_event_detection 
repository version: 1.0 
repository link: https://github.com/jim-schwoebel/acoustic_event_detection 
author: Jim Schwoebel 
author contact: js@neurolex.co 
description: 🎵 A repository for manually annotating files for creating labeled acoustic datasets for machine learning. 
license category: opensource 
license: Apache 2.0 license 
organization name: NeuroLex Laboratories, Inc. 
location: Seattle, WA 
website: https://neurolex.ai 
release date: 2019-04-26 

This code (acoustic_event_detection) is hereby released under a Apache 2.0 license license. 

For more information, check out the license terms below. 

================================================ 
                LICENSE TERMS                      
================================================ 

Copyright 2019 NeuroLex Laboratories, Inc. 
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
                SERVICE STATEMENT                    
================================================ 

If you are using the code written for a larger project, we are 
happy to consult with you and help you with deployment. Our team 
has >10 world experts in Kafka distributed architectures, microservices 
built on top of Node.js / Python / Docker, and applying machine learning to 
model speech and text data. 

We have helped a wide variety of enterprises - small businesses, 
researchers, enterprises, and/or independent developers. 

If you would like to work with us let us know @ develop@neurolex.co. 
'''

###########################################################
## 		    			Import statement 		         ##
###########################################################

import os, librosa, shutil, json, natsort, librosa, os
import librosa.display, sed_vis, dcase_util
from pydub import AudioSegment
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

###########################################################
## 		    			Settings					     ##
###########################################################

# automatically set defaults if there are none and write Settings.JSON
if 'settings.json' not in os.listdir():
	# overlapping windows
	# True or False, allows for overlapping windows in labeling process
	overlapping=False

	# plot feature 
	# plots data segments when labeling (True or False)
	plot_feature=False

	## probability default 
	# Sets the default probability amount (only useful if probability_labeltype == True) 
	# for each labeled session.
	probability_default = 0.80

	## probability labeltype
	# Allows for you to automatically or manually label files with probability of events occuring. 
	# If True, it is automatic; if False, it's manually annotated.
	probability_labeltype = True

	# time to split labeling sections on 
	# 0.20 = 0.20 seconds (or 20 milliseconds)
	timesplit=0.200

	# visualize feature 
	# visualizes all the labeled events after audio labeling is complete for an audio file
	visualize_feature=True 

	# write all these to .JSON for future use 
	jsonfile=open('settings.json','w')
	data={'overlapping': False,
		  'plot_feature': False,
		  'probability_default': 0.80,
		  'probability_labeltype': True,
		  'timesplit': 0.20,
		  'visualize_feature': True}

	json.dump(data,jsonfile)
	jsonfile.close()

else:
	# load from memory if the file exists. Note the file should exist if you 
	# cloned teh repository 
	g=json.load(open('settings.json'))
	overlapping = g['overlapping']
	plot_feature = g['plot_feature']
	probability_default = g['probability_default']
	probability_labeltype = g['probability_labeltype']
	timesplit=g['timesplit']
	visualize_feature = g['visualize_feature']

###########################################################
## 		    	Helper functions					     ##
###########################################################

# now begin plotting linear-frequency power spectrum 
def plot_spectrogram(filename):
	y, sr = librosa.load(filename)
	plt.figure(figsize=(12, 8))
	D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
	plt.subplot(4, 2, 1)
	librosa.display.specshow(D, y_axis='linear')
	plt.colorbar(format='%+2.0f dB')
	plt.title('Linear-frequency power spectrogram')

	# on logarithmic scale 
	plt.subplot(4, 2, 2)
	librosa.display.specshow(D, y_axis='log')
	plt.colorbar(format='%+2.0f dB')
	plt.title('Log-frequency power spectrogram')

	# Or use a CQT scale
	CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=sr), ref=np.max)
	plt.subplot(4, 2, 3)
	librosa.display.specshow(CQT, y_axis='cqt_note')
	plt.colorbar(format='%+2.0f dB')
	plt.title('Constant-Q power spectrogram (note)')
	plt.subplot(4, 2, 4)
	librosa.display.specshow(CQT, y_axis='cqt_hz')
	plt.colorbar(format='%+2.0f dB')
	plt.title('Constant-Q power spectrogram (Hz)')

	# Draw a chromagram with pitch classes
	C = librosa.feature.chroma_cqt(y=y, sr=sr)
	plt.subplot(4, 2, 5)
	librosa.display.specshow(C, y_axis='chroma')
	plt.colorbar()
	plt.title('Chromagram')

	# Force a grayscale colormap (white -> black)
	plt.subplot(4, 2, 6)
	librosa.display.specshow(D, cmap='gray_r', y_axis='linear')
	plt.colorbar(format='%+2.0f dB')
	plt.title('Linear power spectrogram (grayscale)')

	# Draw time markers automatically
	plt.subplot(4, 2, 7)
	librosa.display.specshow(D, x_axis='time', y_axis='log')
	plt.colorbar(format='%+2.0f dB')
	plt.title('Log power spectrogram')

	# Draw a tempogram with BPM markers
	plt.subplot(4, 2, 8)
	Tgram = librosa.feature.tempogram(y=y, sr=sr)
	librosa.display.specshow(Tgram, x_axis='time', y_axis='tempo')
	plt.colorbar()
	plt.title('Tempogram')
	plt.tight_layout()

	# image file save
	imgfile=filename[0:-4]+'.png'
	plt.savefig(imgfile)
	os.system('open %s'%(imgfile))

	return imgfile

def visualize_sample(hostdir, audiofilename, csvfilename):
	# taken from sed_vis documentation - https://github.com/TUT-ARG/sed_vis
	# thanks Audio Research Group, Tampere University! 
	os.system("python3 %s '%s' '%s'"%(hostdir+'/sed_vis/visualize.py', hostdir+'/processed/'+audiofilename, hostdir+'/processed/'+csvfilename))

def window_labeling(filename, classes, plot_feature,probability_default, probability_labeltype):

	os.system('play %s'%(filename))

	# plot only if requested.
	if plot_feature==True:
		plot_spectrogram(filename)

	for a in range(len(classes)):
		if a == len(classes)-1:
			label_text=label_text+classes[a]+' (%s)? \n'%(str(a))
		elif a == 0:
			label_text=classes[a]+' (%s) or '%(str(a))
		else:
			label_text=label_text+classes[a]+' (%s) or '%(str(a))

	label_text=input(label_text)

	for a in range(len(classes)):
		if label_text == str(a):
			label=classes[a]
			shutil.move(os.getcwd()+'/'+filename, os.getcwd()+'/'+classes[a]+'/'+filename)
			break

	# assume 80% probability + get start/stop from .json data
	if probability_labeltype == True:
		probability = probability_default
	elif probability_labeltype == False:
		probability = input('what is the probability of this event (0.0-1.0).')
		termination=False 
		while termination==False:
			try:
				probability=float(probability)
			except:
				try:
					probability = input('what is the probability of this event (0.0-1.0).')
				except:
					pass
	else: 
		# assume 0.80 if nothing is available
		print('no probability given, assuming 0.80 probability...')
		probability = probability_default

	g=json.load(open(filename[0:-4]+'.json'))
	start=g['start']
	stop=g['end']

	# if label does not exist, we can instantiate with n/a to not mess up visualization
	try:
		print(label)
	except:
		label='n/a'

	return filename, start, stop, label, probability

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

def create_csv(csvfilename, filenames, starts, stops, label_texts, probabilities):
	df = pd.DataFrame({'filename': filenames,
					   'onset': starts,
					   'offset': stops,
					   'event_label': label_texts,
					   'probability': probabilities})
	df.to_csv(csvfilename)

def split_segments(filename, timesplit, overlapping):
    # recommend >0.20 seconds for timesplit 
    hop_length = 512
    n_fft=2048 
    y, sr = librosa.load(filename)
    duration=float(librosa.core.get_duration(y))
    
    #Now splice an audio signal into individual elements of 20 ms and extract
    segnum=round(duration/timesplit)
    deltat=duration/segnum
    timesegment=list()
    time=0

    if overlapping == False:
        # non overlapping serial segments spliced by timesplit
        for i in range(segnum):
                #milliseconds
                timesegment.append(time)
                time=time+deltat*1000

    elif overlapping == True:

        for i in range(segnum):
                # overlapping segments spliced by timesplit
                if i ==0:
                        timesegment.append(time)
                        time=time+deltat*1000
                else:
                        timesegment.append(time)
                        time=time-deltat*1000/2
                        timesegment.append(time)
                        time=time+deltat*1000

    newAudio = AudioSegment.from_wav(filename)
    filelist=list()
    file=filename

    # store time data / startstop in parallel to audio file 
    for i in range(len(timesegment)-1):
        filename=exportfile(newAudio,timesegment[i],timesegment[i+1],file,i)
        jsonfile=open(filename[0:-4]+'.json','w')
        data={'start':timesegment[i]/1000,
        	  'end': timesegment[i+1]/1000}
       	json.dump(data,jsonfile)
        filelist.append(filename)

def find_wavfiles(listdir):
	wavfiles=list()
	for a in range(len(listdir)):
		if listdir[a][-4:]=='.wav':
			wavfiles.append(listdir[a])

	return wavfiles

###########################################################
## 		    			MAIN SCRIPT					     ##
###########################################################

# make processed directory to store audio files 
hostdir=os.getcwd()
try:
	os.mkdir('processed')
except:
	pass

os.chdir('data')
listdir=os.listdir()
curdir=os.getcwd()

classnum=input('how many classes do you want? (leave blank for 2) \n')

while True:
	try:
		if classnum=='':
			classnum=2
		classnum=int(classnum)
		break
	except:
		print('error, cannot recognize input') 
		classnum=input('how many classes do you want? (leave blank for 2) \n')

classes=list()
for i in range(classnum):
	classes.append(input('what is class %s? \n'%(str(i+1))))

# make folders and delete contents if existing
for i in range(len(classes)):

	try:
		os.mkdir(classes[i])
	except:
		shutil.rmtree(classes[i])
		os.mkdir(classes[i])

# assumes all files are all uniquely named and are .WAV files
wavfiles=find_wavfiles(listdir)

# now iterate through all wavfiles 
for i in range(len(wavfiles)):
	# create a unique folder for the wavfile
	foldername=wavfiles[i][0:-4]
	os.mkdir(foldername)
	shutil.move(curdir+'/'+wavfiles[i], curdir+'/'+foldername+'/'+wavfiles[i])
	os.chdir(foldername)
	split_segments(wavfiles[i],timesplit, overlapping)

	# instead of removing file we can store it in a ./data/processed directory
	shutil.move(os.getcwd()+'/'+wavfiles[i], hostdir+'/processed/'+wavfiles[i])

	# create folders for classes (e.g. silence and speech)
	for j in range(len(classes)):
		os.mkdir(classes[j])

	# initiate lists for pandas dataframe 
	filenames=list()
	starts=list()
	stops=list()
	label_texts=list()
	probabilities=list()

	# iterate through all the classes and put snippets in proper labeled folder 
	for j in range(len(classes)):
		# make directory and get all new wav files (split folder)
		listdir=os.listdir()
		wavfiles2=find_wavfiles(listdir)
		wavfiles2=natsort.natsorted(wavfiles2)
		time=0

		# now label these files and put them in the appropriate folder 
		for k in range(len(wavfiles2)):
			filename, start, stop, label_text, probability=window_labeling(wavfiles2[k], classes, plot_feature, probability_default, probability_labeltype)
			filenames.append(wavfiles[i])
			starts.append(start)
			stops.append(stop)
			label_texts.append(label_text)
			probabilities.append(probability)

		# now move all folders into proper class folders (e.g. silence or speech folder)
		curdir2=os.getcwd()
		os.chdir(classes[j])
		listdir=os.listdir()
		wavfiles3=find_wavfiles(listdir)
		for k in range(len(wavfiles3)):
			shutil.move(os.getcwd()+'/'+wavfiles3[k], curdir+'/'+ classes[j]+ '/'+wavfiles3[k])
			shutil.move(hostdir+'/data/'+foldername+'/'+wavfiles3[k][0:-4]+'.json', curdir+'/'+classes[j]+'/'+wavfiles3[k][0:-4]+'.json')
		os.chdir(curdir2)

	# now create an output CSV file
	csvfilename=wavfiles[i][0:-4]+'.csv'
	create_csv(csvfilename, filenames,starts,stops,label_texts,probabilities)
	shutil.move(hostdir+'/data/'+foldername+'/'+csvfilename, hostdir+'/processed/'+csvfilename)

	if visualize_feature== True:
		os.chdir(hostdir+'/processed/')
		visualize_sample(hostdir, wavfiles[i],csvfilename)

	# now go back to host directory and repeat for rest of audio files 
	os.chdir(curdir)

	# now delete all the temp folders (optional)
	shutil.rmtree(foldername)



