# RefMix

RefMix is a Python-based tool that uses Spleeter and librosa to analyze, classify, and automatically adjust and mix user-submitted audio stems to match the balance and spectral characteristics of a reference audio track.

# Features

- Separate a reference track into stems using Spleeter (2, 4, or 5 stems).

- Analyze and extract spectral features from audio using Librosa.

- Classify audio stems into categories like vocals, drums, bass, etc.

- Match user stems to reference stems using spectral similarity.

- Calculate gain and frequency-based adjustments for balanced mixing.

- Apply gains and mix stems into a final .wav file.

- Optional stem splitting without mixing.

# Dependencies
Make sure you have the following installed:
```
pip install numpy librosa soundfile pydub spleeter
```
