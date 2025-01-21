'''WIP'''
import numpy as np
import librosa
from src.chords_templates import MAJOR_CHORDS_TEMPLATES, MINOR_CHORDS_TEMPLATES
from src.chords_templates import MAJOR_CHORDS, MINOR_CHORDS


class FeaturesRecognizer:
    ''' Class for recognizing chords and extracting music features. '''

    def __init__(self, hop_length):
        self.hop_length = hop_length

    @staticmethod
    def get_chord_templates():
        ''' Prepare chord templates in chroma space format. '''
        maj_chords = MAJOR_CHORDS
        min_chords = MINOR_CHORDS
        maj_arr, min_arr = MAJOR_CHORDS_TEMPLATES, MINOR_CHORDS_TEMPLATES

        major = maj_arr / np.linalg.norm(maj_arr, axis=0)
        minor = min_arr / np.linalg.norm(min_arr, axis=0)

        return np.concatenate((major, minor), axis=0), [*maj_chords, *min_chords]

    def recognize_chord(self, y, sr, templates, chords):
        ''' 
        Recognize chord from audio signal. 
        Args:
            y: np.array - audio signal.
            sr: int - sample rate.
            templates: np.array - chord templates.
            chords: list[str] - list of chords.
        Returns:
            detected_chord: str - detected chord.
            sr: int - sample rate.
        '''
        if y.max() - y.min() < 0.01:
            return "silence", sr

        y_trimmed, _ = librosa.effects.trim(y, top_db=10)

        chroma_cq = librosa.feature.chroma_cqt(
            y=y_trimmed, sr=sr, hop_length=self.hop_length, norm=None)

        # Normalize chromogram
        mean_chroma = np.mean(chroma_cq, axis=1)
        chroma_vector = mean_chroma / np.linalg.norm(mean_chroma)

        # Get chord match
        similarities = templates @ chroma_vector
        match = np.argmax(similarities)

        detected_chord = chords[match]

        return detected_chord, sr

    def get_music_features(self, y, sr):
        ''' 
        Extract music features from audio signal. 
        Args:
            y: np.array - audio signal.
            sr: int - sample rate.
        Returns:
            features: dict - dictionary with music features:
                - tempo: maximum tempo
                - energy: RMS normalized (assuming 0.1 max)
                - acousticness: lower centroid -> more acoustic
                - danceability: tempo normalized (assuming 200 bpm max)
                - liveness: higher contrast -> more live
                - valence: higher chroma -> happier sound
        '''
        # Extract basic features
        tempo = librosa.feature.tempo(y=y, sr=sr)  # Tempo
        rms = np.mean(librosa.feature.rms(y=y))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
        chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)

        #  Approximate high-level features:
        features = {
            "tempo": np.max(tempo),
            "energy": rms / 0.1,
            "acousticness": 1 - (spectral_centroid / 5000),
            "danceability": np.max(tempo / 200),
            "liveness": spectral_contrast / 50,
            "valence": np.mean(chroma_stft),
        }

        return features
