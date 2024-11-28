import pandas as pd
import ast
import re
import random
import numpy as np

OUTPUT_PATH = 'data\\MIDI\\dataset\\chordonomicon\\'
DATASET_PREPROCESS_PATH = 'data\\MIDI\\dataset_preprocess\\chordonomicon.csv'

# Read the chordonomicon dataset
df = pd.read_csv(DATASET_PREPROCESS_PATH)

# Categorize the chords ======================================================
categories = {
    'pop': [],
    'metal': [],
    'electronic': [],
    'rock': [],
    'soul': [],
    'punk': [],
    'pop_rock': [],
    'country': [],
    'jazz': [],
    'alternative': [],
    'reggae': [],
    'rap': []
}

# Read the mapping CSV file
chord_relations = pd.read_csv('data\\MIDI\\dataset_preprocess\\chords_mapping.csv')

# Create a dictionary with keys the "chords" and values the "degrees"
chord_degrees = dict(zip(chord_relations['Chords'], chord_relations['Degrees']))
for key, value in chord_degrees.items():
    chord_degrees[key] = ast.literal_eval(value)

# Create a dictionary with keys the "chords" and values the "notes"
chord_notes = dict(zip(chord_relations['Chords'], chord_relations['Notes']))
for key, value in chord_notes.items():
    chord_notes[key] = ast.literal_eval(value)
    
# Remove inversions
df['chords'] = df['chords'].apply(lambda s: ' '.join(re.sub(r"/[^ ]+", "", chord) for chord in s.split()))

# Categorize the chords
for index, row in df.iterrows():
    chords = row['chords']
    genre = row['main_genre']  
    if genre in categories and row['parts'] == 'no':
        categories[genre].append(chords)

# Listas de progresiones randoms de cada genero musical
progresion_list = {genre: random.sample(chords, min(50, len(chords))) for genre, chords in categories.items()}

def progression_code(progression, chord_degrees):
    code_progression = []
    
    for chord in progression:
        
        if chord in chord_degrees:
            degrees = []
            degrees = chord_degrees[chord][:]
            duration = random.choice([0.5, 1.0, 1.5, 2.0])
            degrees.append(duration)
            code_progression.append(np.array(degrees))
    
    return code_progression, len(code_progression)

def progressions_code(progressions, chord_degrees):    
    code_progressions = []
    lenghts = []
    
    for progression in progressions:
        [code_progression, lenght] = progression_code(progression, chord_degrees)        
        
        lenghts.append(lenght)
        code_progressions.append(code_progression)
        
    max_lenght = max(lenghts)    
    progressions = []
    
    for progression in code_progressions: 
        while len(progression) < max_lenght:
            progression.append(np.zeros(13)) 
            
        progression = np.array(progression)  
        progressions.append(progression)    

    progressions = np.array(progressions) 
    return progressions

def save_progressions_to_file(progressions, output_file):
    with open(output_file, 'w') as f:
        for i, progression in enumerate(progressions):
            f.write("progresion " + str(i) + ":\n" + str(progression) + '\n') 

genre_01 = progresion_list['pop']
code_progressions_01 = progressions_code(genre_01, chord_degrees)
save_progressions_to_file(code_progressions_01, OUTPUT_PATH + 'pop_progressions.txt')