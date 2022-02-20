# sound_event_detection
A repository for manually annotating audio files to create labeled datasets for machine learning. 

![](https://media.giphy.com/media/vybWlRniCXzZC/giphy.gif)

## How to get started

I'm assuming you are running this on a Mac computer (this is the only operating system tested).

First, make sure you have installed Python3, FFmpeg, and SoX via [Homebrew](https://brew.sh/):

```
brew install python3 sox ffmpeg
```

Now, clone the repository and install all require dependencies:

```
cd ~
git clone git@github.com:jim-schwoebel/sound_event_detection.git
cd sound_event_detection
pip3 install -r requirements.txt
```

## How to label

Just put audio files in the ./data folder, run label_files.py, and then you're ready to get started labeling! See the video below for a quick view on how this can occur (with as many files in the ./data directory that are there). 

### organizing data
First, put all the audio files in the ./data folder. This will allow for the script to go through all these files and set a window (usually 20 milliseconds) to label these audio files. Note that all the audio files in this folder must be uniquely named (e.g. 1.wav, 2.wav, etc.).

### labeling data 
Run the script with 
```
cd ~
cd sound_event_detection
python3 label_files.py
```

This will then ask you for a few things - like the number of classes. Then, all the files are segmented into windows and you can annotate each file. In the example below, 19 files are created (@ 0.50 second windows for a 10 second speech file). See an example terminal session below.

```
how many classes do you want? (leave blank for 2) 
2
what is class 1? 
silence
what is class 2? 
speech
making fast_0.wav
making fast_1.wav
making fast_2.wav
making fast_3.wav
making fast_4.wav
making fast_5.wav
making fast_6.wav
making fast_7.wav
making fast_8.wav
making fast_9.wav
making fast_10.wav
making fast_11.wav
making fast_12.wav
making fast_13.wav
making fast_14.wav
making fast_15.wav
making fast_16.wav
making fast_17.wav
making fast_18.wav

fast_0.wav:

 File Size: 16.0k     Bit Rate: 257k
  Encoding: Signed PCM    
  Channels: 1 @ 16-bit   
Samplerate: 16000Hz      
Replaygain: off         
  Duration: 00:00:00.50  

In:100%  00:00:00.50 [00:00:00.00] Out:22.0k [      |      ]        Clip:0    
Done.
silence (0) or speech (1)?  0
```

After you finish annotating the file, the windowed events are then automatically sorted into the right folders (in the ./data/ directory). In this case, the 0.50 second serial snippets are in the 'speech' and 'silence' directory - all from 1 file (fast.wav). If you had multiple audio files, all the audio file windows would be sorted into these folders to easily prepare these files for machine learning.

![](https://github.com/jim-schwoebel/acoustic_event_detection/blob/master/sed_vis/visualizers/Screen%20Shot%202019-04-26%20at%2011.33.31%20AM.png)

What results is a .CSV annotation file for the entire length of the session in the ./processed/ folder along with the base audio file (e.g. 'fast.wav'). See below for the example annotation. This annotation is necessary for visualizing the file later (the 0.80 probability here can be changed in the settings.json to other values). 

```
filename	onset	offset	event_label	probability
fast.wav	0	0.5	silence	0.8
fast.wav	0.5	1	speech	0.8
fast.wav	1	1.5	speech	0.8
fast.wav	1.5	2	speech	0.8
fast.wav	2	2.5	speech	0.8
fast.wav	2.5	3	speech	0.8
fast.wav	3	3.5	speech	0.8
fast.wav	3.5	4	speech	0.8
fast.wav	4	4.5	speech	0.8
fast.wav	4.5	5	speech	0.8
fast.wav	5	5.5	speech	0.8
fast.wav	5.5	6	speech	0.8
fast.wav	6	6.5	speech	0.8
fast.wav	6.5	7	speech	0.8
fast.wav	7	7.5	speech	0.8
fast.wav	7.5	8	speech	0.8
fast.wav	8	8.5	speech	0.8
fast.wav	8.5	9	speech	0.8
fast.wav	9	9.5	speech	0.8
```

### changing default settings 
You can change a few settings with the SETTINGS.JSON file. Note that for most speech recognition problems, a good window for humans to hear and annotate is 0.20 seconds (or 200 milliseconds), which is the default window used in this repository.

| Setting (Variable)   | Description  | Possible values     |  Default value     |
| ------------- | ---------- | ----------- | ----------- |
| overlapping  | Determines whether or not to use overlapping windows for splicing. | True or False | False |
| model_feature | models data in the timesplit variable + plots onto .CSV file output (for the visualize_feature visualization) | True or False | True | 
| plot_feature | Allows for the ability to plot spectrograms while labeling (8 visuals). | True or False | False |
| probability_default | Sets the default probability amount (only useful if probability_labeltype == True) for each labeled session. | 0.0-1.0 | 0.80 | 
| probability_labeltype | Allows for you to automatically or manually label files with probability of events occuring. If True, the probability event metric is automatically computed with the probability_default value; if False, the probability event metric is manually annotated by the user. | True or False | True |
| timesplit | The window to splice audio by for object detection. If random splicing, the audio will randomly select an interval between 0.20 and 1 seconds (allows for data augmentation). | 0.20-60 or "random" | 0.20 |
| visualize_feature | Allows for the ability to plot events after labeling each audio file. | True or False | False |

## Using machine learning models

### training machine learning models from labels 
You can train a machine learning model easily by running the train_audioTPOT.py script.

```
cd ~
cd sound_event_detection
python3 train_audioTPOT.py
```

You will then be prompted for a few things:

```
Is this a classification (c) or regression (r) problem? --> c
How many classes do you want to train? --> 2 
What is the name of class 1? --> silence
What is the name of class 2? --> speech
```

After this, all the audio files will be featurized with the librosa_featurizing embedding and modeled using [TPOT](https://epistasislab.github.io/tpot/), an AutoML package. Note that much of this code base is from the [Voicebook repository: chapter_4_modeling](https://github.com/jim-schwoebel/voicebook/tree/master/chapter_4_modeling). In this scenario, 25% of the data is left out for cross-validation. 

A machine learning model is then trained on all the data provided in each folder in the ./data directory. Note that if you properly named the classes with label_files.py, then the classes should align (e.g. if you labeled two classes, speech and silence, you can train two classes, silence and speech). 

### making predictions on new files 
You can then easily deploy this machine learning model on new audio files using the load_audioTPOT script.



### applying pre-trained models
If instead you'd like to use some pre-trained models, you can use the ones included in the ./models directory. Here is an overview of all the current models and their accuracies.

Note many of these are overfitted on small datasets, so use these models at your own risk!! :) 

## Visualizing labels and predictions
 
We can use a third-party library called [sed_vis](https://github.com/TUT-ARG/sed_vis) (MIT licensed) to visualize annotated files. I've created a modification script that uses argv[] to pass through the .CSV file label and the audio file so that it works in this interface.

To visualize the files, all you need to do is place the audio file in the ./data folder (and assuming you already have a labeled file known as test.csv with an audio file test.wav - these will be generate with label.py), you can run

```
cd ~
cd sound_event_detection
python3 ./sed_vis/visualize.py ./processed/test.wav ./processed/test.csv
```

What will result will be a visualization like this with all the annotated sound events.

![](https://github.com/jim-schwoebel/acoustic_event_detection/blob/master/sed_vis/visualizers/Figure_1.png)

You can just change the command slightly to visualize all the machine learning models in the ./models directory as well. All you need to do is change the .CSV reference here (e.g. usually it's filename_2.csv):
```
cd ~
cd sound_event_detection
python3 ./sed_vis/visualize.py ./processed/test.wav ./processed/test_2.csv"
```

![](https://github.com/jim-schwoebel/acoustic_event_detection/blob/master/sed_vis/visualizers/Figure_2.png)

With this machine learning visualization, you can better hear how machine learning models are under- or over-fitted and augment datasets, as necessary, for machine learning training.

## Other resources 
If you're interested to learn more about voice computing, I highly encoursge you to check out thie [Voicebook repository](https://github.com/jim-schwoebel/voicebook). This repo contains 200+ open source scripts to get started with voice computing.

Here are some other libraries that may be of interest to learn more about sound event detection:
* [auditok](https://github.com/amsehili/auditok#play-back-detections) 
* [pyannotate-core](https://github.com/pyannote/pyannote-core)
* [PSDS_eval](https://github.com/audioanalytic/psds_eval)
* [pauses](https://github.com/jim-schwoebel/pauses)
* [sed_vis](https://github.com/TUT-ARG/sed_vis)
* [TPOT](https://epistasislab.github.io/tpot/)
* [youtube-scrape](https://github.com/jim-schwoebel/youtube_scrape)
