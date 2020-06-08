# Accent Style Transfer

DSC160 Data Science and the Arts - Final Project - Generative Arts - Spring 2020

Project Team Members: 
- Luis Diaz, lmd003@ucsd.edu
- Catherine Hou, cahou@ucsd.edu
- Prithviraj Pahwa, pspahwa@ucsd.edu
- David Thierry, dthierry@ucsd.edu


### Project Results are also displayed on [This Website](https://sway.office.com/DjvzxcODoEgnft4v?ref=Link)

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

For our project we used NVIDIA's Mellotron as our model. Mellotron is a multispeaker voice synthesis model that can make a voice emote and sing without emotive or singing training data. The GitHub has two pre-trained models (LibriTTS and WaveGlow) that we used to perform our style transfer. The model itself uses outside methods like Denoiser from Waveglow and Tacotron2 while also creating its own plotting / audio processing methods. It requires that each audio file be in .wav format, mono, and at a 24kHz sampling rate. Interestingly, it also uses CMU Pronouncing Dictionary that maps words to its ARPAbet symbol set for speech recognition. That way when the model trains it is able to map a sound to an ARPAbet symbol. There are three key parts of the source audio that the model uses: mel spectogram (which is style transferred into the synthesized audio), the pitch contour (to make the pitch consistent in the synthesized audio), and the rhythm (to evaluate whether the rhythm of the source and synthesized are aligned). Then a denoiser is used to clean up parts of the audio before being returned as a synthesized output.

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
5. LibriTTS dataset
  * The model we used from Mellotron was trained on this dataset. We only used a sample of 100 English speakers, and of those samples we used speakers that had at least 20 minutes of audio. The format of the data path|text|speakerid. All of the files used can be found under mellotron/filelists.
  * https://research.google/tools/datasets/libri-tts/


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
#### Mellotron Accents and Tacotron
[Mellotron Accents and Tacotron](code/MellotronAccents&TacotronTTS.ipynb) This notebook worked on accent conversion on Russian speakers and also using the [Tacotron model](https://github.com/NVIDIA/tacotron2/tree/fc0cf6a89a47166350b65daa1beaa06979e4cddf) also used by NVIDIA 
#### Mellotron Music
[Mellotron Music](code/MellotronMusic.ipynb) This notebook was used when testing the Mellotron on music systhesis on the Hallelujah song on the Mellotron code base

## Results

### To view results in a better format where you can play audio, please visit [this Website](https://sway.office.com/DjvzxcODoEgnft4v?ref=Link) where we present our findings. 

First we generated audio to Handel's Hallelujah as described in the Mellotron code base with different parameters. In the first cell, there is only 1 voice devoted to each of the 4 parts. In the second cell, there are 4 voices devoted to each of the 4 parts.In the third cell we ran the hallelujah with 16 speakers again, but this time with the denoiser. 
- [Hallelujah song by 4 generated speakers with 1 speaker on each part.](/mellotron/custom_results/handel_hallelujah_1_speaker_with_music.wav)
- [Hallelujah song by 16 generated speakers with 4 speakers on each part.](/mellotron/custom_results/handel_hallelujah_4_speaker_with_music.wav)
- [Hallelujah song with 16 speakers denoised](/mellotron/custom_results/denoise_hallelujah.wav)

We sought to use a different input that wasn't used by the Mellotron before, in the Eminem song "Lose Yourself". In the first clip, we displayed the original song without any edits. The second clip is "Lose Yourself" synthesized with a different male speaker using the Mellotron. The third clip was "Lose Yourself" style transferred with a female speaker
- [Original eminem audio](/mellotron/custom_data/eminem00.wav)
- [Style Transferred Eminem audio](/mellotron/custom_results/eminem_synth.wav)
- [Female Style Transferred Eminem audio](/mellotron/custom_results/eminem_female_5s_synth.wav)

In the following section we played audio using the following and then tried to use Mellotron to change the accent of those original speakers. The other speakers are non-native English speakers  and most of the transferred accents seem to be American.

Speech Text:

"Please call Stella.  Ask her to bring these things with her from the store:  Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob.  We also need a small plastic snake and a big toy frog for the kids.  She can scoop these things into three red bags, and we will go meet her Wednesday at the train station."
- [Original Russian Speaker audio](/mellotron/custom_data/russian6.wav)
- [Transferred Russian Speaker to another Female Audio](/mellotron/custom_results/russian6_synth.wav)

Next we looked to see how our Russian clip would work on a male voice. We chose 2 male voices: Stewart Wills and Anders Lankford. As we don't have data regarding the location of the speakers, we can only use from their voices that the 2 voices are American and Scandinavian. We can notable differences between the two speakers. 
- [Transferred Russian Speaker to American Accent Male Audio](/mellotron/custom_results/malerussian6_1.wav)
![Russian](/mellotron/custom_figures/male_russian_1_spectogram.png)

- [Transferred Russian Speaker to Scandinavian Accent Male Audio](/mellotron/custom_results/malerussian6_2.wav)
![Transferred Russian](/mellotron/custom_figures/male_russian_2_spectogram.png)

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
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;We believe accent style transfers to be culturally innovative in multiple aspects of everyday life. With our work, we have found that we can apply and remove different types of “accents” such as regional, variability, and gender accents to different situations. In the first example, we have demonstrated the ability to generate different versions of a song based on certain characteristics. In our Hallelujah example presented, we were able to compose three different versions of the song. The first output presented has only one voice devoted to each of the four parts of the song. In the second output presented there are four voices devoted to each of the four parts. Each of the outputs had a distinct sound to them which could be enjoyed just to the distinct nature of a familiar song. In the Eminem example, we demonstrated the ability to take the original audio vocal sample and then change the voice from male to female speaker. In a social sense, the differences between male and female voices make it such that different speakers can affect how a message conveys a message. This ability to change the speaker of the audio can allow us to imagine statements of things from a different perspective. The third example demonstrates the ability to apply different cultural accents to an audio voice sample. Having the ability to take any voice sample and apply an accent allows people to understand how different a language is to another. Each of these examples share the common characteristic of allowing us to reimagine the audio as we currently know it. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;With the ability to apply different types of accents to different audio samples, we have the potential for a vast amount of sociolinguistic research opportunities. With the application of male and female style transfer, we can study how differently the two types of audio perform with certain sales tasks in telemarketing and customer outreach which can then be used to improve business operations. This type of work can also be used to help us better understand why certain types of AI assistants (Siri, Alexa, etc.) come with multiple voice options and why users predominately choose one type.  With the cultural style transfer applications, we can utilize the style transfer methods presented to assist others with learning other languages. Learning a new language isn’t easy but knowing that you are saying a phrase correctly given your accent can be reassuring. This can also be used for efficient work between culturally diverse teams. Exposure to accents prior to a conversation can help both sides better prepare for verbal quirks they were not expecting. As for political issues, this work can potentially be used for such types of issues such as helping people get their voice out and understanding more information. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;There are multiple potential ethical concerns that can arise from our application on style transfer on audio samples. These ethical issues mainly present themselves when dealing with the Male/Female style transferred audio example and the primary language style transfer example. In the former style transfer method, there is research to show that female voices are considered more helpful and compliant whereas male voices are considered more commanding and assertive. Because of these characteristics associated with male and female, applying a male or female style to an audio sample could potentially be used for manipulation or for changing the intent of a message. As for the Primary Language style transfer method, we believe that ethical issues affect individuals who reside in or around the areas where the accent originates. For example, applying a region’s accent can come off as offensive when shown the result because the intentions behind these works may be misinterpreted as trying to stereotype them. It is also very possible that the regional accent applied misrepresents due to the data we used in our methods, which can offend. These two methods presented both present ethical issues regarding manipulation and representation, which is why it is important to utilize tools such as these for research.  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This work can be expanded in many ways. For instance, we could extend our work by training our own model using Mellotron’s methods. However because each instance in Kaggle’s Speech Accent Archive is of the same text, we would not have enough variety of pronunciations to support a new model. If we had access to the right kind of dataset (where we have an adequate amount of audio from a speaker with a regional accent and the text associated with), we could hopefully apply regional accents to American accents. We can also try to improve the clarity of our outputs and experiment with different variations of “accents”. In regards to the first example with Hallelujah, we could instead try to recreate the song using other instrument sounds instead of human sounds. This would recreate the song as it would be if it was created using a different instrument. In the example containing the Eminem song “Lose Yourself”, we could expand the idea of applying a female voice to a male (or the reverse case) artist’s songs to recreating an album of an artist with their counterpart voice. However, there is more potential in regards to the regional accent example, where we were able to apply the Male American Accent to the Spanish Accent. These methods can be used to make a platform that takes audio which has an accent and converts the audio into a form that is more easily understandable to the user based on their linguistic background. For example, two product development teams who share a similar secondary language must work together to meet a deadline. This work can be expanded such that we can remove an accent from an audio source in real-time. This would allow for seamless communication between the two diverse teams and would increase efficiency. However, this task may be difficult because obtaining the spoken training data for a task of that size would be difficult.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Finally, our approach regarding the works on this project differs from traditional generative methods today. With today’s methods, accent transformation is typically achieved by using Automated Speech Recognition to transcribe accented speech. This transcribed accent speech is then converted back into speech with Text-to-Speech synthesis. This method typically presents challenges when dealing with audio with heavy accents or incomplete speaker vocabulary.


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

- Our Website: https://sway.office.com/DjvzxcODoEgnft4v?ref=Link
