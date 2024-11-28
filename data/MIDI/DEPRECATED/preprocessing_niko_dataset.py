import pickle
import numpy as np

OUTPUT_PATH = 'data\\MIDI\\dataset\\niko_dataset\\'
DICTIONARY_PATH='data\\MIDI\\dataset\\note_dict.json'

# Cargar el dataset
with open('data\\MIDI\\dataset_preprocess\\MIDI\\03_niko_sataset_(MIDI)\\Niko Dataset\\dataset.pkl', 'rb') as file:
    dataset = pickle.load(file)

#print(dataset)

#{'piece name': 
#   {'nmat': [[0, 3, 60, 60], ...], # 2-d matrix: note matrix [start, end, pitch, velocity]
#   'root': [[0,0,0,0,0,0,0,0], ...], # 2-d matrix: root label 
#   'style': 'some style', # pop_standard, pop_complex, dark, r&b, unknown 
#   'mode': 'some mode', # M, m 
#   'tonic': 'C' # C, Db, ..., B }, ... }
    

# Variables para almacenar las progresiones según su género
pop_standard = []
pop_complex = []
dark = []
rnb = []
unknown = []

# Variables para almacenar las progresiones según el modo
major = []
minor = []

# Función para codificar una progresión de acordes
def encode_chord_progression(piece):   # pieza ---> progresion de acordes de la pieza 
    encoded_progression = []
    current_chord = []
    current_start = None

    for note in piece['nmat']: # itera en cada nota de la pieza
        start, _, _, _ = note
        if current_start is None:
            current_start = start

        # el ciclo va acumular notas en la variable current_chord hasta que la variable start 
        # sea diferente a current_start, y en ese momento codificará el acorde actual y lo añadirá 
        # a encoded_progression
        
        if start != current_start:
            # Codificar el acorde actual
            encoded_chord = [0] * 12
            for n in current_chord:
                encoded_chord[n[2] % 12] = 1  # Convertir pitch a rango 0-11
            encoded_chord.append(current_chord[0][1] - current_start)  # Duración del acorde
            encoded_progression.append(np.array(encoded_chord))

            # Reiniciar para el siguiente acorde
            current_chord = []
            current_start = start
    
        current_chord.append(note) # --> añade la nota al acorde actual --> [start, end, pitch, velocity]

    # Codificar el último acorde
    if current_chord:
        encoded_chord = [0] * 12
        for n in current_chord:
            encoded_chord[n[2] % 12] = 1  # Convertir pitch a rango 0-11
        encoded_chord.append(current_chord[0][1] - current_start)  # Duración del acorde
        encoded_progression.append(encoded_chord)

    return encoded_progression

# Iterar sobre cada pieza en el dataset

encoded_progressions = []
lengths = []
for piece_name, piece_data in dataset.items():
    
    encoded_progression = encode_chord_progression(piece_data)
    lengths.append(len(encoded_progression))
    
    genre = piece_data['style']
    mode = piece_data['mode']

    
    encoded_progressions.append(encoded_progression)
    
    # Ordenar las progresiones según su género
    if genre == 'pop_standard':
        pop_standard.append(encoded_progression)
    elif genre == 'pop_complex':
        pop_complex.append(encoded_progression)
    elif genre == 'dark':
        dark.append(encoded_progression)
    elif genre == 'r&b':
        rnb.append(encoded_progression)
    else:
        unknown.append(encoded_progression)
        
    # Ordenar las progresiones según su género
    if mode == 'M':
        major.append(encoded_progression)
    elif mode == 'm':
        minor.append(encoded_progression)

# Convetir las progresiones en arrays de numpy
progressions_final = []
for progression in encoded_progressions:
    max_length = max(lengths)
    while len(progression) < max_length:
        progression.append(np.zeros(13))
    progressions_final.append(np.array(progression))

progressions_final = np.array(progressions_final)

print(progressions_final.shape)
# Función para guardar las progresiones en un archivo
def save_progressions_to_file(progressions, output_file):
    """Guardar las progresiones armónicas en un archivo de texto"""
    with open(output_file, 'w') as f:
        for i, progression in enumerate(progressions):
            f.write("progresion " + str(i) + ":\n" + str(progression) + '\n')


if __name__ == '__main__':
    
    # Guardar las progresiones rnb en un archivo
    save_progressions_to_file(progressions_final, OUTPUT_PATH + 'niko_dataset_all.txt')

    # Guardar las progresiones según su género y modo
    for genre, progressions in zip(['pop_standard', 'pop_complex', 'dark', 'r&b', 'unknown'], [pop_standard, pop_complex, dark, rnb, unknown]):
        save_progressions_to_file(progressions, OUTPUT_PATH + 'by_genre\\' + 'niko_dataset_' + genre + '.txt')
        
    for mode, progressions in zip(['Major', 'Minor'], [major, minor]):
        save_progressions_to_file(progressions, OUTPUT_PATH + 'by_mode\\' + 'niko_dataset_' + mode + '.txt')