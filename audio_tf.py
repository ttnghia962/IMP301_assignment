from pydub import AudioSegment
from pydub.playback import play
from pydub import AudioSegment


def happy_filter(input_file, output_file, treble_boost, bass_cut, compressor_ratio, compressor_attack, compressor_release):
    # Đọc file âm thanh đầu vào
    sound = AudioSegment.from_mp3(input_file)

    # Tăng cường âm treble và hạ bass
    sound = sound + treble_boost
    sound = sound - bass_cut

    # Áp dụng hiệu ứng compressor
    sound = sound.compress_dynamic_range(ratio=compressor_ratio, attack=compressor_attack, release=compressor_release)

    # Ghi âm thanh đã lọc ra file mới
    sound.export(output_file, format="mp3")

    print("Equalizer and RAP filter applied successfully!")

if __name__ == "__main__":
    # Thay đổi đường dẫn file input và output tùy thuộc vào nhu cầu của bạn
    input_file = "E:/IMP301_assignment/wd.mp3"
    output_file = "E:/IMP301_assignment/output/wd_filtered.mp3"

    treble_boost = 10  # dB
    bass_cut = 3  # dB
    compressor_ratio = 8.0  # Tỷ lệ nén của compressor
    compressor_attack = 10  # Thời gian attack của compressor (ms)
    compressor_release = 200  # Thời gian release của compressor (ms)

    # apply_equalizer(input_file, output_file, treble_boost, bass_cut)
    happy_filter(input_file, output_file, treble_boost, bass_cut, compressor_ratio, compressor_attack, compressor_release)
