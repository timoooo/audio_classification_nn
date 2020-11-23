# THIS IS *Work in progress*

Fooling around with pytorch and nn. 
- GPU is automatically detected and used.
- images are generated by myself (will maybe compare vs Kaggle Images in the future) audio_data#generateImages()
    - it will automatically DELETE existing images folder and create new images.

Source test data: https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification

# TO-DO

- Compare image recognition vs directly feeding the data in the nn (eg mean of features) 
- image augumentation? --> inc acc?
- audio data augumentation? (manipulate distortion & pitches)--> inc acc?
- image augumenation vs audio data augumentation   what is more efficient?
- try torch audio to load the data

#Results
NN v1 Spectrogram Testresult: 29%   --> Does increase in features result in more accurate recognitions? increase features by a lot for v2

NN v2 Spectrogram Testresult: 21% --> Accuarcy slightly decreased but computation time increased a lot. --> fewer feature set more epochs to train

NN v5 Spectrogram Testresult 41% --> lowering the lr increased acc by a lot

NN v6 Spectrogram Testresult 50% --> by far the best results with this setup   
