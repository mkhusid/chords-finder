'''
Implementation of the ChordsAnalizer class for recording and analyzing chords.
Important third-party packages:
    - pyaudio - provides Python bindings for PortAudio, the cross-platform audio I/O library.
    - librosa - provides a Python interface for music and audio analysis.
'''
import sys
import io
import warnings
import wave
import csv
from datetime import datetime
import numpy as np
import pyaudio
import librosa
from src.features_recognizer import FeaturesRecognizer

warnings.filterwarnings("ignore")


BUFF_SIZE = 1024 * 8  # Increased buffer size
CHANNELS = 1 if sys.platform == "darwin" else 2
SAMPLE_RATE = 44100


class ChordsAnalizer:
    ''' Class for recording and analyzing chords.
        Constructor parameters:
            recognizer: FeaturesRecognizer - object for recognizing chords.
            buff_size: buffer size for audio stream.
            channels: number of channels for audio stream.
            sample_rate: sample rate of the input audio stream.
            smooth_interval: interval for smoothing transitions between chords,
                by excluding unexpected chroma jumps (related to noise).
    '''

    def __init__(self, recognizer: FeaturesRecognizer, buff_size,
                 channels, sample_rate, smooth_interval=5):
        self.buff_size = buff_size
        self.channels = channels
        self.sample_rate = sample_rate
        self.smooth_interval = smooth_interval
        self.writer = None
        self.recognizer = recognizer
        self.templates, self.chords = recognizer.get_chord_templates()
        self.recorded_data = bytes()
        self.rows = []
        self.record_name = None
        self.initialize_audio_stream()

    def initialize_audio_stream(self):
        ''' Initialize audio stream. '''
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.buff_size,
        )
        return self.stream

    def _write_chords(self, chords_list: list[str], counter: int,
                      frames_list: list[int], filtered_chords: list[str],
                      user_name: str, record_name: str):
        '''
        Write chords to the output file. 
        Args:
            chords_list: list[str] - list of chords.
            counter: int - counter for chords.
            frames_list: list[int] - list of frames.
            filtered_chords: list[str] - list of filtered chords.
            user_name: str - user name.
            record_name: str - record name.
        '''

        chunk_data = self.stream.read(self.buff_size, exception_on_overflow=False)
        int_data = np.frombuffer(chunk_data, dtype=np.int16) / 32768.0
        counter['frame'] += 1

        chord, sr = self.recognizer.recognize_chord(
            int_data, self.sample_rate, self.templates, self.chords)

        if chord != 'silence' and chord == chords_list[-1]:
            counter["repeates"] += 1
            counter['current_chord'] = chord

        if chord != 'silence' and chord != chords_list[-1]:
            if counter["repeates"] > self.smooth_interval:
                filtered_chords.append(chords_list[-1])
                frames_list.append(counter["frame"])
                print(counter)

            chords_list.append(chord)

            if len(filtered_chords) > 0 and len(filtered_chords) % 3 == 0:
                frames = np.array(frames_list)
                timeframes = librosa.frames_to_time(frames, sr=sr, hop_length=self.buff_size)
                new_row = {
                    'timestamp': datetime.now().strftime('%m/%d/%y %H:%M:%S'),
                    'chords': str(filtered_chords),
                    'frames': [f'{time:.2f}' for time in timeframes],
                    'user_name': user_name,
                    "record_name": record_name
                }
                self.rows.append(new_row)
                # print('Output file updated with new chords.')
            counter["repeates"] = 1

        self.recorded_data += chunk_data

    def record_chords(self, csv_file: io.StringIO, user_name, record_name):
        ''' 
        Record chords and write them to the output file. 
        Args:
            csv_file: io.StringIO - file to write chords to.
            user_name: str - user name.
            record_name: str - record name.
        '''
        print("Detecting chords:")

        chords_list, filtered_chords = [''], []
        frames_list = []

        chord_counter = {"current_chord": '', "repeates": 0, "frame": 0}

        fieldnames = ['timestamp', 'chords', 'frames', 'user_name', 'record_name', 'features']
        self.writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        self.writer.writeheader()
        self.rows = []
        self.record_name = record_name

        while self.stream.is_active():
            self._write_chords(
                chords_list, chord_counter, frames_list,
                filtered_chords, user_name, record_name
            )

    def save_record(self):
        ''' Save final CSV and audio for the record. '''
        # pylint: disable=no-member
        output_path_wav = f"./recorded/{self.record_name}.wav"
        with wave.open(output_path_wav, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.pa.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.sample_rate)
            wf.writeframes(self.recorded_data)

        y, sr = librosa.load(output_path_wav)
        features = self.recognizer.get_music_features(y, sr)

        for row in self.rows:
            row['features'] = features
        self.writer.writerows(self.rows)

        return output_path_wav

    def build_final_chroma(self, path):
        ''' Build final chroma for the record. '''
        y, sr = librosa.load(path)
        y_trimmed, _ = librosa.effects.trim(y, top_db=10)
        return librosa.feature.chroma_cqt(y=y_trimmed, sr=sr)

    def stop_audio_stream(self):
        ''' Close audio stream '''
        output_wav = self.save_record()

        self.stream.close()
        self.stream.stop_stream()
        self.pa.terminate()
        print('Audio stream is closed. File with chords was saved succesfully.')
        return output_wav


def main():
    ''' Main function for recording and analyzing chords. '''
    chords_recognizer = FeaturesRecognizer(hop_length=256)
    chords_analyzer = ChordsAnalizer(chords_recognizer, BUFF_SIZE, CHANNELS, SAMPLE_RATE)

    record_name = input('Please specify record name: ')
    user_name = input('Please specify your user_name: ')
    with open(f'./streamed_chords/{record_name}.csv', 'w+', encoding='utf-8') as csv_output:
        chords_analyzer.record_chords(csv_output, user_name, record_name)


if __name__ == "__main__":
    main()
