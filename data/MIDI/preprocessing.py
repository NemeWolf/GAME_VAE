from preprocessing_chordonomicon import ChordonomiconPreprocessor
from preprocessing_nikodataset import NikodatasetPreprocessor
from preprocessing_free_midi import MIDIProcessor
import numpy as np

if __name__ == '__main__':
    
    # Preprocess Chordonomicon dataset
    
    MAPPING_PATH = 'data/MIDI/dataset_preprocess/chords_mapping.csv'
    DATASET_PATH = 'data/MIDI/dataset_preprocess/chordonomicon.csv'
    OUTPUT_PATH = 'data/MIDI/dataset/chordonomicon/'

    preprocessor_chordonomicon = ChordonomiconPreprocessor(
        dataset_path=DATASET_PATH,
        mapping_path=MAPPING_PATH,
        output_path=OUTPUT_PATH,
        sample_amount=2000
    )
    preprocessor_chordonomicon.call()
    progresions_chordonomicon = preprocessor_chordonomicon.progressions_genre
    
    # Preprocess Nikodataset
    
    DATASET_PATH = 'data\\MIDI\\dataset_preprocess\\MIDI\\03_niko_sataset_(MIDI)\\Niko Dataset\\dataset.pkl'
    OUTPUT_PATH = 'data\\MIDI\\dataset\\niko_dataset\\'

    preprocesor_nikodataset = NikodatasetPreprocessor(
        dataset_path=DATASET_PATH,
        output_path=OUTPUT_PATH
    )
    preprocesor_nikodataset.call()
    
    pop_standard = preprocesor_nikodataset.pop_standard
    pop_complex = preprocesor_nikodataset.pop_complex
    dark = preprocesor_nikodataset.dark
    rnb = preprocesor_nikodataset.rnb
    unknown = preprocesor_nikodataset.unknown
    major = preprocesor_nikodataset.major
    minor = preprocesor_nikodataset.minor
    final_progressions = preprocesor_nikodataset.final_progressions 
    
    # Preprocess Free MIDI dataset
    
    OUTPUT_PATH = 'data\\MIDI\\dataset\\free_midi_2024\\chords_progresions.txt'
    DICTIONARY_PATH='data\\MIDI\\dataset\\note_dict.json'
    MIDI_PATH = 'data\\MIDI\\dataset_preprocess\\MIDI\\PROGRESSIONS\\05_free-midi-progressions-20241011_(MIDI)\\'

    preocessor_midi = MIDIProcessor(
        output_path=OUTPUT_PATH,
        dictionary_path=DICTIONARY_PATH,
        midi_path=MIDI_PATH
    )

    # processor.create_mapping()  # Uncomment if you need to create the mapping
    major, minor, modal = preocessor_midi.process_by_category()
    
    input_major, target_major = preocessor_midi.create_dataset(major)
    input_minor, target_minor = preocessor_midi.create_dataset(minor)
    input_modal, target_modal = preocessor_midi.create_dataset(modal)
      
    preocessor_midi.save_progressions_to_file(major, OUTPUT_PATH+'major.txt')
    preocessor_midi.save_progressions_to_file(minor, OUTPUT_PATH+'minor.txt')
    preocessor_midi.save_progressions_to_file(modal, OUTPUT_PATH+'modal.txt')

    print(f"Input shape: {input_major.shape}, Target shape: {input_major.shape}")
    print(f"Input shape: {input_minor.shape}, Target shape: {input_minor.shape}")
    print(f"Input shape: {input_modal.shape}, Target shape: {input_modal.shape}")