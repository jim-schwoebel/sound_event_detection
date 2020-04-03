import os, json, time
import pandas as pd 
import speech_recognition as sr_audio

# transcribe with pocketsphinx (open-source)
def transcribe_sphinx(file):

	try:
		r=sr_audio.Recognizer()
		with sr_audio.AudioFile(file) as source:
			audio = r.record(source) 
		transcript=r.recognize_sphinx(audio)
		print('sphinx transcript: '+transcript)
	except:
		print('error')
		transcript=''
		
	return transcript 

listdir=os.listdir()
print(listdir)
jsonfiles=list()

for i in range(len(listdir)):
	if listdir[i].endswith('.json'):
		jsonfiles.append(listdir[i])

keywords=list()
paths=list()
detects=list()
times=list()

for i in range(len(jsonfiles)):
	g=json.load(open(jsonfiles[i]))
	keywords.append(g['keyword'])
	paths.append(g['audio_path'])
	detects.append(g['detect'])
	times.append(g['time'])

	if g['detect'] == False:
		# os.system('play %s'%(g['audio_path'].split('/')[-1][0:-6]+'.wav'))
		# time.sleep(5)
		transcribe_sphinx(g['audio_path'].split('/')[-1][0:-6]+'.wav')
		time.sleep(5)

data={'keywords': keywords,
	  'paths': paths,
	  'detects': detects,
	  'times': times}

print(data)
df=pd.DataFrame(data,columns=list(data))
df.to_csv('data.csv')