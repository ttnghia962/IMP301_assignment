import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def treble_boost_equalizer(y, sr, boost_factor=2.0):
    # Tính toán Fourier Transform của tín hiệu âm thanh
    D = librosa.stft(y)
    
    # Lấy amplitudes và pha của tần số
    mag, phase = librosa.magphase(D)
    
    # Tăng cường amplitudes của tần số cao (treble)
    mag_high_boosted = mag * boost_factor
    
    # Tạo một ma trận mới từ amplitudes đã tăng cường và pha ban đầu
    D_high_boosted = mag_high_boosted * np.exp(1j * phase)
    
    # Tính toán tín hiệu mới từ biến đổi ngược Fourier
    y_high_boosted = librosa.istft(D_high_boosted)
    
    return y_high_boosted

# Đọc file âm thanh
input_audio_path = 'path/to/your/audio/file.mp3'
y, sr = librosa.load(input_audio_path)

# Áp dụng Equalizer
y_boosted = treble_boost_equalizer(y, sr)

# Hiển thị biểu đồ tín hiệu âm thanh gốc và đã tăng cường
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
librosa.display.waveshow(y, sr=sr)
plt.title('Original Audio')

plt.subplot(2, 1, 2)
librosa.display.waveshow(y_boosted, sr=sr)
plt.title('Treble Boosted Audio')

plt.tight_layout()
plt.show()
