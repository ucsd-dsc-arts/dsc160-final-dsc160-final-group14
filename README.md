# Accent Style Transfer

DSC160 Data Science and the Arts - Final Project - Generative Arts - Spring 2020

Project Team Members: 
- Luis Diaz, lmd003@ucsd.edu
- Catherine Hou, cahou@ucsd.edu
- Prithviraj Pahwa, pspahwa@ucsd.edu
- David Thierry, dthierry@ucsd.edu


Project Results are also displayed on __[This Website]__(https://sway.office.com/DjvzxcODoEgnft4v?ref=Link)

## Abstract

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

### Model 

For our project we used NVIDIA's Mellotron as our model. Mellotron is a multispeaker voice synthesis modelthat can make a voice emote and sing without emotive or singing training data.

* [NVIDIA's Mellotron](https://github.com/NVIDIA/mellotron)
* [Mellotron: Multispeaker expressive voice synthesis by conditioning on rhythm, pitch and global style tokens](https://arxiv.org/abs/1910.11997)


### Data

We gathered data from different sources to test on our model. Some data, like the hallelujah music clip, were in the Mellotron github already. Data sources are listed below:

1. The Speech Accent Archive
  * This dataset includes 2140 unique speech samples of the same passage. The individuals who participated and performed the speech recordings come from a total of 177 countries. There are 214 unique native languages encompassed within the countries. We only used a few of these samples.
  * https://www.kaggle.com/mfekadu/english-multispeaker-corpus-for-voice-cloning
2.  WavSource.com
  * From here we gathered some clips and audio made by famous people to use as well
  * http://www.wavsource.com/people/famous.htm
3.  MegaWavs.com

  * audio clips from invader zim were gathered from here
  * http://www.megawavs.com/cartoon-sounds.aspx?title=Invader+Zim&qty=28
4.  freesoundeffects.com
  * audio clip for a camel were gathered from here
  * https://www.freesoundeffects.com/free-sounds/camels-10019/


## Code

TBecause we used [NVIDIA's mellotron](https://github.com/NVIDIA/mellotron) as our code base, we followed the steps on their [README](https://github.com/NVIDIA/mellotron/blob/master/README.md) to set up the pre-trained model. In otder to run the notebooks  the following code should be run ('git clone https://github.com/NVIDIA/waveglow.git') to download the [Waveglow](https://github.com/NVIDIA/waveglow/tree/2fd4e63e2918012f55eac2c8a8e75622a39741be) files. To set up the Mellotron [this Libritts](https://drive.google.com/open?id=1ZesPPyRRKloltRIuRnGZ2LIUEuMSVjkI) file and this [Waveglow file](https://drive.google.com/open?id=1okuUstGoBe_qZ4qUEF8CcwEugHP7GM_b) had to be downloaded and placed in a new directory called 'models'. Data that we used had to be manually downloaded from kaggle or the other sources and potentially changed from mp3 to wav files but we placed those files in our [custom_data](https://github.com/ucsd-dsc-arts/dsc160-final-dsc160-final-group14/tree/master/mellotron/custom_data) directory. 

### Rhythm and Pitch Transfer
#### Processing
[Audio Concatenation](mellotron/audio-concatenation.ipynb) Because the model is used on subsets of audio, this notebook strings all the synthesized audio clips together into one wav file. <br>
[Audio Trimming](mellotron/audio-trimming.ipynb) This notebook separates the wav file into subsets to make processing easier.
#### Eminem's Lose Yourself
[Eminem's Lose Yourself Inference](mellotron/inference-eminem.ipynb) This notebook loads the pre-trained libritts model, filelists, 
dataholders, speakers and performs the transfer on one of the speakers from the dataset.
#### Russian Accent
[Russian Inference](mellotron/inference-russian6.ipynb) This notebooks does the same but for a [russian accent](mellotron/custom_data/russian6.wav) on one of the speakers from the dataset.
#### Spanish Accent
[Spanish Inference](mellotron/inference-spanish100.ipynb) This notebooks does the same but for a [spanish accent](mellotron/custom_data/spanish100.wav) on one of the speakers from the dataset. 
#### Famous Speakers
[Famous Speaker Inference](mellotron/famous_speaker_inference.ipynb) This notebook applies a style transfer on several famous people, as well as the cartoon character, Invader Zim, and on a camel
#### Famous Speakers
[Mellotron Accents and Tacotron](code/MellotronAccents&TacotronTTS.ipynb) This notebook worked on accent conversion on Russian speakers and also using the [Tacotron model](https://github.com/NVIDIA/tacotron2/tree/fc0cf6a89a47166350b65daa1beaa06979e4cddf) also used by NVIDIA 
#### Famous Speakers
[Mellotron Music](code/MellotronMusic.ipynb) This notebook was used when testing the Mellotron on music systhesis on the Hallelujah song on the Mellotron code base

## Results

(30 points) 

This section should summarize your results and will embed links to documentation to significant outputs. This should document both process and show artistic results. This can include figures, sound files, videos, bitmaps, as appropriate to your generative art idea. Each result should include a brief textual description, and all should be listed below: 

- image files (`.jpg`, `.png` or whatever else is appropriate)
- audio files (`.wav`, `.mp3`)
- written text as `.pdf`

To view results in a better format where you can play audio, please visit __[this Website]__(https://sway.office.com/DjvzxcODoEgnft4v?ref=Link) where we present our findings. 

First we generated audio to Handel's Hallelujah as described in the Mellotron code base with different parameters. In the first cell, there is only 1 voice devoted to each of the 4 parts. In the second cell, there are 4 voices devoted to each of the 4 parts.In the third cell we ran the hallelujah with 16 speakers again, but this time with the denoiser. 
- [Hallelujah song by 4 generated speakers with 1 speaker on each part.](/mellotron/custom_results/handel_hallelujah_1_speaker_with_music.wav)
- [Hallelujah song by 16 generated speakers with 4 speakers on each part.](/mellotron/custom_results/handel_hallelujah_4_speaker_with_music.wav)
- [Hallelujah song with 16 speakers denoised](/mellotron/custom_results/denoise_hallelujah.wav)

We sought to use a different input that wasn't used by the Mellotron before, in the Eminem song "Lose Yourself". In the first clip, we displayed the original song without any edits. The second clip is "Lose Yourself" synthesized with a different male speaker using the Mellotron. The third clip was "Lose Yourself" style transferred with a female speaker
- [Original eminem audio](/mellotron/custom_data/eminem00.wav)
- [Style Transferred Eminem audio](/mellotron/custom_results/eminem_synth.wav)
- [Female Style Transferred Eminem audio]()

In the following section we played audio using the following and then tried to use Mellotron to change the accent of those original speakers. The other speakers are non-native English speakers  and most of the transferred accents seem to be American.

Speech Text:

"Please call Stella.  Ask her to bring these things with her from the store:  Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob.  We also need a small plastic snake and a big toy frog for the kids.  She can scoop these things into three red bags, and we will go meet her Wednesday at the train station."
- [Original Russian Speaker audio](/mellotron/custom_data/russian6.wav)
- [Transferred Russian Speaker to another Female Audio](/mellotron/custom_results/russian6_synth.wav)

Next we looked to see how our Russian clip would work on a male voice. We chose 2 male voices: Stewart Wills and Anders Lankford. As we don't have data regarding the location of the speakers, we can only use from their voices that the 2 voices are American and Scandinavian. We can notable differences between the two speakers. 
- [Transferred Russian Speaker to American Accent Male Audio](/mellotron/custom_results/malerussian6_1.wav)
![Russian](/mellotron/custom_figures/eminem00.png)

- [Transferred Russian Speaker to Scandinavian Accent Male Audio](/mellotron/custom_results/malerussian6_2.wav)
![Transferred Russian](/mellotron/custom_figures/eminem01.png)

We attempted further analysis with speakers of with different native languages. This is the result with a native Spanish speaker. The results were not as good as the our Russian speakers, but we can still make out the words.
- [Original Spanish Accent Audio](/mellotron/custom_data/spanish100.wav)
- [Transferred Female Spanish Speaker to Male American Accent](/mellotron/custom_results/spanish100_synth.wav)

Furthermore we then tried to this style transfer to famous speakers, cartoon characters, and even animals where the audio was transferred to a different voice. However none of the results were clear and sounded like gibberish.Here are some of the results that we played around with to see how they would come out. Several of the results did not make coherent speech and did not work as we expected.
- [Original Clint Eastwood audio](/mellotron/custom_data/eastwood_lawyers_1.wav)
- [Transferred Clint Eastwood audio](/mellotron/custom_results/clint_eastwood.wav)


- [Original MLK audio](/mellotron/custom_data/king_war_no_more_1.wav)
- [Transferred MLK audio](/mellotron/custom_results/MLK.wav)


- [Original Invader Zim audio](/mellotron/custom_data/invader_zim_1.wav)
- [Transferred Invader Zim audio](/mellotron/custom_results/invader_zim.wav)


- [Original Camel audio](/mellotron/custom_data/camel6_1.wav)
- [Transferred Camel audio to human](/mellotron/custom_results/camel.wav	)

We thought the following was interest because since Stephen Hawking uses a speech synthesizer to speak we thought that it would be easier to generate good results. However, the results were not very clear
- [Original Stephen Hawking audio](/mellotron/custom_data/hawking02_1.wav)
- [Transferred Stephen Hawking audio](mellotron/custom_results/Hawking.wav)

We also played around with the Tacotron2 and Waveglow models to generate iconic quotes in new voices. In particular, we wish to transform famous political quotes to a new, female voice.

The first block has the following quote by John F Kennedy:

"Those who make peaceful revolution impossible will make violent revolution inevitable."
- [JFK Quote](mellotron/custom_results/JKF_quote_female.wav)

The following block has the famous post Pearl Harbor Speech by Franklin D Roosevelt:

"Yesterday, December 7th, 1941 -- a date which will live in infamy -- the United States of America was suddenly and deliberately attacked by naval and air forces of the Empire of Japan. The United States was at peace with that nation and, at the solicitation of Japan, was still in conversation with its government and its emperor looking toward the maintenance of peace in the Pacific."
- [FDR Quote](mellotron/custom_results/FDR_quote_female.wav)

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

- Luis Diaz: Worked on the website and on writing several parts of the report on the README. Also worked on the MEllotron model and on accent conversion on speakers audio clips in the famous_speaker_inference notebook. 
- Catherine Hou: Worked on the Mellotron model and on the accent conversion of several speakers, on the eminem music, and work done on the audio-concatenation, audio-trimming, inference-eminem, inference-russian6, and inference-spanish100 notebooks
- Prithviraj Pahwa: Worked on the Mellotron model and on the accent conversion of several speakers. Also created examples using the tacotron to use text to speech to produce results. Worked on the MellotronAccents&TacotronTTS and MellotronMusic notebooks.
- David Thierry: Worked on the README and on the discussion of our projects. Also worked on the website

## Technical Notes and Dependencies

- To run the project on datahub, all the packages and necessary imports are in the [requirements.txt](/mellotron/requirements.txt)
that can be installed using 'pip install -r requirements.txt'. To run on google colab all the packages in the requirements.txt still need to be pip installed as well as these two packages: 'unidecode' and 'tensorflow==1.13.1'

## Reference

- Accent Conversion Using Artificial Neural Networks: https://pdfs.semanticscholar.org/e362/207b67aa1f6dbf5ea2d9e01edeeda70ba15e.pdf

- Self-imitating Feedback Generation Using GAN for Computer-Assisted Pronunciation Training: https://arxiv.org/ftp/arxiv/papers/1904/1904.09407.pdf

- Accent Classification and Neural Accent Transfer: http://cs230.stanford.edu/files_winter_2018/projects/6939642.pdf

- Audio texture synthesis and style transfer: https://dmitryulyanov.github.io/audio-texture-synthesis-and-style-transfer/

- An open source implementation of Neural Voice Cloning with Few Samples: https://sforaidl.github.io/Neural-Voice-Cloning-With-Few-Samples/

- NVIDIA's Mellotron: https://github.com/NVIDIA/mellotron

- NVIDIA's Waveglow https://github.com/NVIDIA/waveglow/tree/2fd4e63e2918012f55eac2c8a8e75622a39741be

- Mellotron- Multispeaker expressive voice synthesis by conditioning on rhythm, pitch and global style tokens: https://arxiv.org/abs/1910.11997

- Kaggle dataset: https://www.kaggle.com/mfekadu/english-multispeaker-corpus-for-voice-cloning

- Famous Speaker audio clips: http://www.wavsource.com/people/famous.htm

- Invader Zim audio clips: http://www.megawavs.com/cartoon-sounds.aspx?title=Invader+Zim&qty=28

- Camel audio clip: https://www.freesoundeffects.com/free-sounds/camels-10019/
