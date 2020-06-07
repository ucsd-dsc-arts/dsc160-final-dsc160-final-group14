# Accent Style Transfer

DSC160 Data Science and the Arts - Final Project - Generative Arts - Spring 2020

Project Team Members: 
- Luis Diaz, lmd003@ucsd.edu
- Catherine Hou, cahou@ucsd.edu
- Prithviraj Pahwa, pspahwa@ucsd.edu
- David Thierry, dthierry@ucsd.edu


Project Results are also displayed on [This Website](https://sway.office.com/V39N7sFoHz5r66Bj?ref=Link)

## Abstract

(10 points) 

For the project proposal, please write a short abstact addressing the questions below. You need to replace the entire contents of this section with one to two paragraphs addressing the following:

- What is your concept for a generative art project? 
- What methods/networks/techniques will you employ (include links to technical precedents/code bases)
- What training data (if any) will you use for your project? 
- What kind of results do you hope that your system will produce?
- How will you present your result/what form will your output take?
- What if any challenges to you think may arise as you are working with this?
- How are you expanding on topics we have covered in class? 
- Why is it interesting? (personally, culturally, politically, other)
- List three papers / art projects that are references for this work.

### ABSTRACT V1
  Our goal is to take an audio voive sample and apply a style transfer that can apply an accent to the voice. Some previous work in accent style transfer has been researched using [Neural Networks](https://pdfs.semanticscholar.org/e362/207b67aa1f6dbf5ea2d9e01edeeda70ba15e.pdf). For our project we plan to try and transfer accents using GANS, possibly cycle GANS with some techniques described in this [paper](https://arxiv.org/ftp/arxiv/papers/1904/1904.09407.pdf). We plan to use training data from [this kaggle dataset](https://www.kaggle.com/mfekadu/english-multispeaker-corpus-for-voice-cloning) of English Multi-Speakers and from [this](https://www.kaggle.com/rtatman/speech-accent-archive) speech accent archive. 
  
  We will output audio samples of similar lengths with the new accents applied to them. This will then be described and presented in a report of our results with potential for a website where people can apply audio to add an accent to. Some challenges that we think will arise are using some of the tools due to our unfailiarity and then applying accents might have unexpected results especially in our short time frame. We have discussed style transfer and generative techniques in lecture, here we will apply to audio data and see if we can change accents of the speaker. This is interesting because we can see how voices can be reimagined in a different speaker voice.
  
### ABSTRACT V2
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The concept of this generative art project is to apply style transfer methods on audio files to produce an augmented version of the original audio file. Specifically, we plan to take an audio voice sample and apply a style transfer to return that sample as a new audio file where the audio presents a different accent than it had before. For our project we plan to try and transfer accents using Generative Adversarial Networks. We also are considering the use of cycle GANS with some more advanced techniques described in this [paper](https://arxiv.org/ftp/arxiv/papers/1904/1904.09407.pdf).  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The training data we will use in this generative art project will come from two different sources on Kaggle.com. The first dataset will be the English Multi-speaker Corpus for Voice Cloning dataset found here. The second dataset will be the Speech Accent Archive dataset found here. From the combination of these two datasets as training data, we want our model to output audio samples of input similar lengths with different accents applied to them. The augmented audio files which will be returned will be mp3 files. The user of this generative art project should be able to input an mp3 file of a certain length and the output should be that same audio file with an accent applied to it. Some challenges that we expect to arise while completing this assignment include group unfamiliarity with style transfer techniques applied to audio can lead to mistakes in implementation and as a result of the former our results may our may not be what we are expecting. In other words, s due to our unfamiliarity and then applying accents might have unexpected results, especially in a short time frame. We have discussed style transfer and generative techniques in lecture, here we will apply to audio data and see if we can change the accents of the speaker. We believe that the Accent Style Transfer was a particularly interesting project idea because of the prevalence of accents in human language. If we are able to apply an accent to an audio source effectively, this provides a foundation for analyzing how accents in language affect interpretation and message conveyance. 

### References 
Source 1: [Accent Conversion Using Artificial Neural Networks
](https://pdfs.semanticscholar.org/e362/207b67aa1f6dbf5ea2d9e01edeeda70ba15e.pdf)

Source 2: [Self-imitating Feedback Generation Using GAN
for Computer-Assisted Pronunciation Training](https://arxiv.org/ftp/arxiv/papers/1904/1904.09407.pdf)

Source 3: [Accent Classification and Neural Accent Transfer](http://cs230.stanford.edu/files_winter_2018/projects/6939642.pdf)

Source 4: [Audio texture synthesis and style transfer](https://dmitryulyanov.github.io/audio-texture-synthesis-and-style-transfer/)

Source 5: [An open source implementation of Neural Voice Cloning with Few Samples](https://sforaidl.github.io/Neural-Voice-Cloning-With-Few-Samples/)


## Data and Model

(10 points) 

In the final submission, this section will describe both the data you use for this project and any pre-existing models/neural nets. For each you should provide the name, a textual description, and a link. If there is a paper (for neural net) link that as well.
- Such and such Neural Net. The short description of this neural net. 
  - [link to code]().
  - [Title of Paper with Link](). 
- Training data. Short description of training data including bibliographic info. [link to data]().

### Data 

The English Multi-speaker Corpus for Voice Cloning
- This dataset includes speech data from 109 native English speakers with varying types of vocal accents.  Each of the speakers in this data set was asked to read out 400 sentences where each sentence is from a set of sentences designed to maximize the unique contextual and phonetic of the English language. The speech data in this dataset was recorded with “an omnidirectional head-mounted microphone, 96kHz sampling frequency at 24 bits, and in a Hemi-anechoic chamber of the University of Edinburgh.”
- https://www.kaggle.com/mfekadu/english-multispeaker-corpus-for-voice-cloning

The Speech Accent Archive
- This dataset includes 2140 unique speech samples of the same passage. The individuals who participated and performed the speech recordings come from a total of 177 countries. Of these 174 countries, there are 214 unique native languages encompassed within the countries. It was not mentioned how the audio for each unique sample was recorded, unlike the English Multi-speaker Corpus for Voice Cloning dataset we previously mentioned. 
- https://www.kaggle.com/mfekadu/english-multispeaker-corpus-for-voice-cloning 

### Model 

TODO


## Code

(20 points)

This section will link to the various code for your project (stored within this repository). Your code should be executable on datahub, should we choose to replicate your result. This includes code for: 

- code for data acquisition/scraping
- code for preprocessing
- training code (if appropriate)
- generative methods

Link each of these items to your .ipynb or .py files within this seection, and provide a brief explanation of what the code does. Reading this section we should have a sense of how to run your code.

### Rhythm and Pitch Transfer
#### Processing
[Audio Concatenation](mellotron/audio-concatenation.ipynb) Because the model is used on subsets of audio, this notebook strings all the synthesized audio clips together into one wav file.
[Audio Trimming](mellotron/audio-trimming.ipynb) This notebook separates the wav file into subsets to make processing easier.
#### Eminem's Lose Yourself
[Eminem's Lose Yourself Inference](mellotron/inference-eminem.ipynb) This notebook loads the pre-trained libritts model, filelists, 
dataholders, speakers and performs the transfer on one of the speakers from the dataset.
#### Russian Accent
[Russian Inference](inference-russian6.ipynb) This notebooks does the same but for a [russian accent](mellotron/custom_data/russian6.wav) on one of the speakers from the dataset.
#### Spanish Accent
[Russian Inference](inference-russian6.ipynb) This notebooks does the same but for a [spanish accent](mellotron/custom_data/spanish100.wav) on one of the speakers from the dataset. 

## Results

(30 points) 

This section should summarize your results and will embed links to documentation to significant outputs. This should document both process and show artistic results. This can include figures, sound files, videos, bitmaps, as appropriate to your generative art idea. Each result should include a brief textual description, and all should be listed below: 

- image files (`.jpg`, `.png` or whatever else is appropriate)
- audio files (`.wav`, `.mp3`)
- written text as `.pdf`

## Discussion

(30 points, three to five paragraphs)

The first paragraph should be a short summary describing your results.

The subsequent paragraphs could address questions including:
- Why is this culturally innovative?
- How does your generative computational approach differ from traditional art/music/cultural production? 
- How do your results relate to broader social, cultural, economic political, etc., issues? 
- What are the ethical concerns for this form of generative art? 
- In what future directions could you expand this work?

## Team Roles

Provide an account of individual members and their efforts/contributions to the specific tasks you accomplished.

## Technical Notes and Dependencies

Any implementation details or notes we need to repeat your work. 
- Additional libraries you are using for this project
- Does this code require other pip packages, software, etc?
- Does this code need to run on some other (non-datahub) platform? (CoLab, etc.)

## Reference

All references to papers, techniques, previous work, repositories you used should be collected at the bottom:
- Papers
- Repositories
- Blog posts
