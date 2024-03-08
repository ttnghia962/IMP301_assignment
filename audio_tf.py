from pydub import AudioSegment
from pydub.playback import play
from pydub import AudioSegment


def apply_audio_filter(input_file, output_file, treble_boost, bass_cut, compressor_ratio, compressor_attack, compressor_release, stat):
    sound = AudioSegment.from_mp3(input_file)

    sound = sound + treble_boost
    sound = sound - bass_cut

    sound = sound.compress_dynamic_range(ratio=compressor_ratio, attack=compressor_attack, release=compressor_release)

    output_file_save= output_file + stat + '_filtered.mp3'
    sound.export(output_file_save, format="mp3")
    print(f"Filter applied successfully to {output_file_save}!")


def happy_filter(input_file, output_file):
    apply_audio_filter(input_file, output_file, treble_boost=10, bass_cut=3, compressor_ratio=2, compressor_attack=10, compressor_release=100, stat='happy' )

def fear_filter(input_file, output_file):
    apply_audio_filter(input_file, output_file,  treble_boost=2, bass_cut=5, compressor_ratio=3, compressor_attack=20, compressor_release=150,  stat='fear')

def neutral_filter(input_file, output_file):
    apply_audio_filter(input_file, output_file,  treble_boost=0, bass_cut=0, compressor_ratio=1, compressor_attack=5, compressor_release=50, stat='neutral')

def scare_filter(input_file, output_file):
    apply_audio_filter(input_file, output_file, treble_boost=2, bass_cut=7, compressor_ratio=2.5, compressor_attack=15, compressor_release=120, stat='scare')

def sad_filter(input_file, output_file):
    apply_audio_filter(input_file, output_file, treble_boost=-3, bass_cut=-2, compressor_ratio=1.5, compressor_attack=8, compressor_release=80, stat='sad')



if __name__ == "__main__":
    # Thay đổi đường dẫn file input và output tùy thuộc vào nhu cầu của bạn
    input_file = "E:/IMP301_assignment/wd.mp3"
    output_file = "E:/IMP301_assignment/output/"


    # user= input("Choice your type: Happy, Scare, Sad, Fear, Neutral filters")
 

    with open('txt_file_path.txt', 'r') as file:
        content = file.read()
        
        arr= content.split()
        if "Happy" == arr[1]:
            happy_filter(input_file, output_file)
        elif "Scare" == arr[1]:
            scare_filter(input_file, output_file)
        elif "Sad" == arr[1]:
            sad_filter(input_file, output_file)
        elif "Fear" == arr[1]:
            fear_filter(input_file, output_file)
        elif "Neutral" == arr[1]:
            neutral_filter(input_file, output_file)

