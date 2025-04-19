import numpy as np
import soundfile as sf

# Create a 2-second silent audio file
sample_rate = 16000
duration = 2.0
audio = np.zeros(int(sample_rate * duration))
sf.write('examples/test_audio.wav', audio, sample_rate)
print('Created test audio file: examples/test_audio.wav') 