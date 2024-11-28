import pickle
import numpy as np

DATASET_PATH = 'data\\MIDI\\dataset_preprocess\\MIDI\\03_niko_sataset_(MIDI)\\Niko Dataset\\dataset.pkl'
OUTPUT_PATH = 'data\\MIDI\\dataset\\niko_dataset\\'

class NikodatasetPreprocessor:
    def __init__(self, dataset_path, output_path):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.dataset = self.load_dataset()
        self.pop_standard = []
        self.pop_complex = []
        self.dark = []
        self.rnb = []
        self.unknown = []
        self.major = []
        self.minor = []
        self.encoded_progressions = []
        self.lengths = []
        self.final_progressions = []

    def call(self):
        self.process_dataset()
        self.processing_all_progressions()
        self.save_all_progressions()
        
    def load_dataset(self):
        with open(self.dataset_path, 'rb') as file:
            return pickle.load(file)

    def encode_chord_progression(self, piece):
        encoded_progression = []
        current_chord = []
        current_start = None

        for note in piece['nmat']:
            start, _, _, _ = note
            if current_start is None:
                current_start = start

            if start != current_start:
                encoded_chord = [0] * 12
                for n in current_chord:
                    encoded_chord[n[2] % 12] = 1
                encoded_chord.append(current_chord[0][1] - current_start)
                encoded_progression.append(np.array(encoded_chord))

                current_chord = []
                current_start = start

            current_chord.append(note)

        if current_chord:
            encoded_chord = [0] * 12
            for n in current_chord:
                encoded_chord[n[2] % 12] = 1
            encoded_chord.append(current_chord[0][1] - current_start)
            encoded_progression.append(encoded_chord)

        return encoded_progression

    def process_dataset(self):
        for piece_name, piece_data in self.dataset.items():
            encoded_progression = self.encode_chord_progression(piece_data)
            self.lengths.append(len(encoded_progression))
            genre = piece_data['style']
            mode = piece_data['mode']

            self.encoded_progressions.append(encoded_progression)

            if genre == 'pop_standard':
                self.pop_standard.append(encoded_progression)
            elif genre == 'pop_complex':
                self.pop_complex.append(encoded_progression)
            elif genre == 'dark':
                self.dark.append(encoded_progression)
            elif genre == 'r&b':
                self.rnb.append(encoded_progression)
            else:
                self.unknown.append(encoded_progression)

            if mode == 'M':
                self.major.append(encoded_progression)
            elif mode == 'm':
                self.minor.append(encoded_progression)

    def convert_dataset_numpy(self, preogressions):
        progressions_final = []
        max_length = max(self.lengths)
        
        for progression in preogressions:
            while len(progression) < max_length:
                progression.append(np.zeros(13))
            progressions_final.append(np.array(progression))
        progressions_final = np.array(progressions_final)    
               
        return progressions_final
    
    def processing_all_progressions(self):
        self.final_progressions = self.convert_dataset_numpy(self.encoded_progressions)
        self.pop_standard = self.convert_dataset_numpy(self.pop_standard)
        self.pop_complex = self.convert_dataset_numpy(self.pop_complex)
        self.dark = self.convert_dataset_numpy(self.dark)
        self.rnb = self.convert_dataset_numpy(self.rnb)
        self.unknown = self.convert_dataset_numpy(self.unknown)
        self.major = self.convert_dataset_numpy(self.major)
        self.minor = self.convert_dataset_numpy(self.minor)
        
        print(self.final_progressions.shape)

    def save_progressions_to_file(self, progressions, output_file):
        with open(output_file, 'w') as f:
            for i, progression in enumerate(progressions):
                f.write("progresion " + str(i) + ":\n" + str(progression) + '\n')

    def save_all_progressions(self):        
                
        self.save_progressions_to_file(self.final_progressions, self.output_path + 'niko_dataset_all.txt')

        for genre, progressions in zip(['pop_standard', 'pop_complex', 'dark', 'r&b', 'unknown'], 
                                       [self.pop_standard, self.pop_complex, self.dark, self.rnb, self.unknown]):
            self.save_progressions_to_file(progressions, self.output_path + 'by_genre\\' + 'niko_dataset_' + genre + '.txt')

        for mode, progressions in zip(['Major', 'Minor'], [self.major, self.minor]):
            self.save_progressions_to_file(progressions, self.output_path + 'by_mode\\' + 'niko_dataset_' + mode + '.txt')

if __name__ == '__main__':
    processor = NikodatasetPreprocessor(
        dataset_path=DATASET_PATH,
        output_path=OUTPUT_PATH
    )
    processor.call()