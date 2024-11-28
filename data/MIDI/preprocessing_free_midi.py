import music21 as m21
import numpy as np
import os
import json

OUTPUT_PATH = 'data\\MIDI\\dataset\\free_midi_2024\\chords_progresions.txt'
DICTIONARY_PATH='data\\MIDI\\dataset\\note_dict.json'
MIDI_PATH = 'data\\MIDI\\dataset_preprocess\\MIDI\\PROGRESSIONS\\05_free-midi-progressions-20241011_(MIDI)\\'

class MIDIProcessor:
    def __init__(self, output_path, dictionary_path, midi_path):
        self.output_path = output_path
        self.dictionary_path = dictionary_path
        self.midi_path = midi_path
        self.note_dict = self.load_note_dict(dictionary_path)

    def load_midi(self, file_path):
        """Convierte de manera segura un archivo MIDI en un objeto music21"""
        return m21.converter.parse(file_path)

    def extract_chords(self, midi_file):
        """Extraer acordes de objeto Music21 filtrando las notas repetidas y ordenando las notas del acorde"""
        chords = []
        for element in midi_file.flatten().notes:
            if isinstance(element, m21.chord.Chord):
                unique_pitch_classes = set(p.pitchClass for p in element.pitches)
                unique_pitches = [m21.pitch.Pitch(p) for p in unique_pitch_classes]
                unique_chord = m21.chord.Chord(unique_pitches)
                unique_chord.sortAscending()
                duration = element.quarterLength
                chords.append([unique_chord, duration])
        return chords

    def process_midi_files(self, midi_path):
        """Procesar todos los archivos MIDI en el directorio dataset_path"""
        progressions = []
        unique_progressions = set()

        for dirpath, dirnames, filenames in os.walk(midi_path):
            for file in filenames:
                if file.endswith('.mid'):
                    file_path = os.path.join(dirpath, file)
                    midi_file = self.load_midi(file_path)
                    chords = self.extract_chords(midi_file)
                    progression = []
                    for c, t in chords:
                        chord_vector = [0] * 12
                        for p in c.pitches:
                            index = self.note_dict[p.name]
                            chord_vector[index] = 1
                        chord_vector.append(t)
                        progression.append(np.array(chord_vector))
                    progression_tuple = tuple(map(tuple, progression))
                    if progression and progression_tuple not in unique_progressions:
                        unique_progressions.add(progression_tuple)
                        progressions.append(np.array(progression))

        if progressions:
            max_len = max(len(p) for p in progressions)
            padded_progressions = []
            for p in progressions:
                padded_progression = np.zeros((max_len, 13))
                padded_progression[:len(p), :] = p
                padded_progressions.append(padded_progression)
            return np.array(padded_progressions)
        else:
            return np.array([])

    def process_by_category(self):
        major_path = self.midi_path + 'Major'
        minor_path = self.midi_path + 'Minor'
        modal_path = self.midi_path + 'Modal'
        
        major_progressions=self.process_midi_files(major_path)
        minor_progressions=self.process_midi_files(minor_path)
        modal_progressions=self.process_midi_files(modal_path)
        
        return major_progressions, minor_progressions, modal_progressions
    
    def save_progressions_to_file(self, progressions, output_path):
        """Guardar las progresiones armónicas en un archivo de texto"""
        with open(output_path, 'w') as f:
            for i, progression in enumerate(progressions):
                f.write("progresion " + str(i) + ":\n" + str(progression) + '\n')

    def create_mapping(self):
        """Crea y guarda un diccionario con las notas y sus codigos"""
        note_dict = {}
        for i in range(12):
            note_dict[m21.pitch.Pitch(i).name] = i
        with open(self.dictionary_path, 'w') as f:
            json.dump(note_dict, f)

    def load_note_dict(self, dict_path):
        """Cargar un diccionario con las notas y sus codigos"""
        with open(dict_path, 'r') as f:
            note_dict = json.load(f)
        return note_dict

    def create_dataset(self, progressions):
        """Crear el dataset de entrada y salida para el modelo"""
        input = progressions
        target = progressions
        return input, target

if __name__ == '__main__':
    processor = MIDIProcessor(
        output_path=OUTPUT_PATH,
        dictionary_path=DICTIONARY_PATH,
        midi_path=MIDI_PATH
    )

    # processor.create_mapping()  # Uncomment if you need to create the mapping
    major, minor, modal = processor.process_by_category()
    
    input_major, target_major = processor.create_dataset(major)
    input_minor, target_minor = processor.create_dataset(minor)
    input_modal, target_modal = processor.create_dataset(modal)
      
    processor.save_progressions_to_file(major, OUTPUT_PATH+'major.txt')
    processor.save_progressions_to_file(minor, OUTPUT_PATH+'minor.txt')
    processor.save_progressions_to_file(modal, OUTPUT_PATH+'modal.txt')

    print(f"Input shape: {input_major.shape}, Target shape: {input_major.shape}")
    print(f"Input shape: {input_minor.shape}, Target shape: {input_minor.shape}")
    print(f"Input shape: {input_modal.shape}, Target shape: {input_modal.shape}")