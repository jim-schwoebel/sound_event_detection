# acoustic_event_detection
A repository for manually annotating files for creating labeled acoustic datasets for machine learning.

![](https://media.giphy.com/media/vybWlRniCXzZC/giphy.gif)

## How to get started

I'm assuming you are running this on a Mac computer (this is the only operating system tested).

First, make sure you have installed Python3, FFmpeg, and SoX via [Homebrew](https://brew.sh/):

```
brew install python3 sox ffmpeg
```

Now, clone the repository and install all require dependencies:

```
git clone git@github.com:jim-schwoebel/acoustic_event_detection.git
pip3 install -r requirements.txt
```

## How to label

### organizing data
First, put all the audio files in the ./data folder. This will allow for the script to go through all these files and set a window (usually 20 milliseconds) to label these audio files. 

Run the script with 
```
python3 acoustic_event_detection.py
```

This will then ask you for a few things - like the number of classes. 

Then, all the files are segmented into windows and you can annotate each file. 

What results is a .CSV annotation file for the entire length of the session.

### settings 
You can change a few settings with the .JSON file. (show table)

## Training machine learning models

You can train a machine learning model easily by running the train_audioTPOT.py script.

```
python3 train_audioTPOT.py
```

You will then be prompted for a few things:

```
Is this a classification (c) or regression (r) problem? --> c
How many classes do you want to train? --> 2 
What is the name of class 1? --> silence
What is the name of class 2? --> speech
```

After this, all the audio files will be featurized with the librosa_featurizing embedding and modeled using [TPOT](https://epistasislab.github.io/tpot/), an AutoML package. Note that much of this code base is from the [Voicebook repository: chapter_4_modeling](https://github.com/jim-schwoebel/voicebook/tree/master/chapter_4_modeling).

## Visualizing labels and predictions
 
We can use a third-party library called [sed_vis](https://github.com/TUT-ARG/sed_vis) (MIT licensed) to visualize annotated files. I've created a modification script that uses argv[] to pass through the .CSV file label and the audio file so that it works in this interface.

- output visualizations

To visualize the files, all you need to do is run argv[]. 

![](https://github.com/jim-schwoebel/acoustic_event_detection/blob/master/sed_vis/visualizers/Figure_1.png)

## Datasets generated with script

Datasets used: [AudioSet], the [Common Voice Project], [YouTube], and [train-emotions].

### silence vs. speech detection
This dataset is from about 100 speech files and has 1,000 unique events. 

### music events.
1,000 samples.

### silence events.
1,000 samples.

### noise events.
1,000 samples.

### phonemes 
1,000 samples in each category.

## FAQs
* do you have overlapping window options? --> currently, no. But this is something we're working on.

## Future things to do
1. add in sed_vis library for annotation and playback (for scientific publications) 
2. format the annotation process in the format of sed_vis so that any annotation session can be visualized well 
3. be able to prospectively deploy machine learning models to make predictions via sed_vis visualization library (100 ms windows, break up + make predictions, export .CSV and then display predictions vs. actual results). 

## Additional reading
* [SwipesForScience](https://github.com/SwipesForScience/SwipesForScience)
