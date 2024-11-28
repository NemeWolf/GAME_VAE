import pandas as pd
import ast
import re
import random
import numpy as np

MAPPING_PATH = 'data/MIDI/dataset_preprocess/chords_mapping.csv'
DATASET_PATH = 'data/MIDI/dataset_preprocess/chordonomicon.csv'
OUTPUT_PATH = 'data/MIDI/dataset/chordonomicon/'

class ChordonomiconPreprocessor:
    def __init__(self, dataset_path, mapping_path, output_path, sample_amount):
        self.dataset_path = dataset_path
        self.mapping_path = mapping_path
        self.output_path = output_path
        self.categories = {
            'pop': [], 'metal': [], 'electronic': [], 'rock': [], 'soul': [], 'punk': [],
            'pop rock': [], 'country': [], 'jazz': [], 'alternative': [], 'reggae': [], 'rap': []
        }
        self.df = pd.read_csv(self.dataset_path)
        self.chord_degrees = {}
        self.progresion_list = {}
        
        self.sample_amount = sample_amount
        
        self.progressions_genre = {
            'pop': None, 'metal': None, 'electronic': None, 'rock': None, 'soul': None, 'punk': None,
            'pop rock': None, 'country': None, 'jazz': None, 'alternative': None, 'reggae': None, 'rap': None
        }
        
    def call(self):
        self.load_mappings()
        self.remove_inversions()
        self.categorize_chords()
        self.process_and_save_all()

    def load_mappings(self):
        """Load chord mappings from csv file."""
        chord_relations = pd.read_csv(self.mapping_path)
        self.chord_degrees = dict(zip(chord_relations['Chords'], chord_relations['Degrees']))
        for key, value in self.chord_degrees.items():
            self.chord_degrees[key] = ast.literal_eval(value)
        
    def remove_inversions(self):
        """Remove inversions from chords."""
        self.df['chords'] = self.df['chords'].apply(lambda s: ' '.join(re.sub(r"/[^ ]+", "", chord) for chord in s.split()))

    def categorize_chords(self):
        """Categorize chords by genre and return a dictionary with the genres as keys and lists of chords as values."""
        for index, row in self.df.iterrows():
            chords = row['chords']
            genre = row['main_genre']
            if genre in self.categories and row['parts'] == 'no':
                self.categories[genre].append(chords)
        self.progresion_list = {genre: random.sample(chords, min(self.sample_amount, len(chords))) for genre, chords in self.categories.items()}

    def progression_code(self, progression):
        """Convert a chord progression str to a list of chord degrees."""
        code_progression = []
        for chord in progression:
            if chord in self.chord_degrees:
                degrees = self.chord_degrees[chord][:]
                duration = random.choice([0.5, 1.0, 1.5, 2.0])
                degrees.append(duration)
                code_progression.append(np.array(degrees))
        return code_progression, len(code_progression)

    def progressions_code(self, progressions):
        """Convert a list of chord progressions str to a numpy array of chord degrees."""
        code_progressions = []
        lengths = []
        for progression in progressions:
            code_progression, length = self.progression_code(progression)
            lengths.append(length)
            code_progressions.append(code_progression)
        max_length = max(lengths)
        progressions = []
        for progression in code_progressions:
            while len(progression) < max_length:
                progression.append(np.zeros(13))
            progressions.append(np.array(progression))
        return np.array(progressions)

    def process_progressions_by_genre(self, genre):
        genre_progressions = self.progresion_list[genre]
        code_progressions = self.progressions_code(genre_progressions)
        return code_progressions
    
    def save_progressions_to_file(self, progressions, output_file):
        with open(output_file, 'w') as f:
            for i, progression in enumerate(progressions):
                f.write("progresion " + str(i) + ":\n" + str(progression) + '\n')

    def process_and_save_all(self):
        for genre in self.progresion_list.keys():
            code_progressions = self.process_progressions_by_genre(genre)            
            self.progressions_genre[genre] = code_progressions
            
            self.save_progressions_to_file(code_progressions, self.output_path + f'{genre}_progressions.txt')
            print(f'{genre} progressions saved.')          

# main
if __name__ == '__main__':
    preprocessor = ChordonomiconPreprocessor(
        dataset_path=DATASET_PATH,
        mapping_path=MAPPING_PATH,
        output_path=OUTPUT_PATH,
        sample_amount=2000
    )

    preprocessor.call()
