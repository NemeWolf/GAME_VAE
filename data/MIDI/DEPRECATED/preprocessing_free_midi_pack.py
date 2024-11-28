import music21 as m21
import numpy as np
import os
import json

OUTPUT_PATH = 'data\\MIDI\\dataset\\chords_progresions_test.txt'
DICTIONARY_PATH='data\\MIDI\\dataset\\note_dict.json'
#MIDI_PATH = "data\\MIDI\\dataset_preprocess\\MIDI\\PROGRESSIONS\\"

# test
MIDI_PATH = "data\\MIDI\\dataset_preprocess\\MIDI\\PROGRESSIONS\\02_Unison_(Midi)"

# CARGAR ARCHIVOS MIDI =========================================================
def load_midi( file_path ):
    """Convierte de manera segura un archivo MIDI en un objeto music21"""    
    return m21.converter.parse( file_path )

# EXTRAER ACORDES ==============================================================
def extract_chords( midi_file ):
    """Extraer acordes de objeto Music21 filtrando las notas repetidas y ordenando las notas del acorde"""
    chords = []
    for element in midi_file.flatten().notes:
        if isinstance(element, m21.chord.Chord):
            # Eliminar notas repetidas en diferentes octavas
            unique_pitch_classes = set( p.pitchClass for p in element.pitches )
            unique_pitches = [ m21.pitch.Pitch( p ) for p in unique_pitch_classes ]
            unique_chord = m21.chord.Chord( unique_pitches )            
            # Ordenar las notas del acorde desde la mas grave hasta la mas aguda
            unique_chord.sortAscending()    
            # Agregar el acorde y su duracion a la lista de acordes        
            duration = element.quarterLength
            chords.append([ unique_chord,duration ]) # --> [ [ chord, duration ], ... ]
            
    return chords

# PROCESAR ARCHIVOS MIDI =======================================================
def process_midi_files(dataset_path,note_dict):
    """Procesar todos los archivos MIDI en el directorio dataset_patH"""    
    progressions = []
    unique_progressions = set()  # Conjunto para almacenar progresiones únicas

    for dirpath, dirnames, filenames in os.walk( dataset_path ):
        for file in filenames:            
            if file.endswith('.mid'):                
                # Cargar el archivo MIDI
                file_path = os.path.join( dirpath, file )
                midi_file = load_midi( file_path )
                
                # Extraer acordes
                chords = extract_chords( midi_file )                          # --> [ [ <music21.chord.Chord C4 E4 G4>, 0.5 ], ... ]                
                
                # Codificar las progresiones armonicas
                progression = []                
                for c,t in chords:
                    chord_vector = [0] * 12  # Vector de 12 elementos para las notas de DO a SI
                    for p in c.pitches:
                        index = note_dict[p.name]  
                        chord_vector[index] = 1     
                    chord_vector.append(t)                      # --> [ 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0.5 ]
                    progression.append(np.array(chord_vector))  # --> [ [ 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0.5 ], ... ]
                    
                #if progression:  
                #    progressions.append(progression)            # --> [ [ [ 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0.5 ], ... ], ... ]
                
                # Convertir la progresión a una tupla de arrays para verificar unicidad
                progression_tuple = tuple(map(tuple, progression))
                
                # Solo añadir progresiones no vacías y únicas     
                if progression and progression_tuple not in unique_progressions:
                    unique_progressions.add(progression_tuple)  
                    progressions.append(np.array(progression))  # --> [ [ [ 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0.5 ], ... ], ... ]        
                
                # DEPRECATED
                
                # for c,t in chords:                     
                #     chord = [ p.nameWithOctave for p in c.pitches ]           # --> [ 'C4', 'E4', 'G4' ]                
                #     encoded_chord = [ note_dict[ note ] for note in chord ]   # --> [ 0, 4, 7 ]
                #     encoded_chord.append( t )                                 # --> [ 0, 4, 7, 0.5 ]
                #     progression.append( encoded_chord )                       # --> [ [ 0, 4, 7, 0.5], ... ]
                # progressions.append( progression )                        # --> [ [ [ 0, 4, 7, 0.5], ... ], ... ]
                # return progressions
    
    # Rellenar las progresiones para que todas tengan el mismo tamaño
    if progressions:
        max_len = max(len(p) for p in progressions)
        padded_progressions = []
        for p in progressions:
            padded_progression = np.zeros((max_len, 13))  # Crear una matriz de ceros con la forma adecuada
            padded_progression[:len(p), :] = p  # Rellenar con la progresión original
            padded_progressions.append(padded_progression)
        return np.array(padded_progressions)
    else:
        return np.array([])
                
# GUARDAR PROGRESIONES EN UN ARCHIVO DE TEXTO ==================================
def save_progressions_to_file( progressions, output_file ):
    """Guardar las progresiones armónicas en un archivo de texto"""    
    with open( output_file, 'w' ) as f:
        for i,progression in enumerate( progressions ):
            f.write( "progresion "+str( i )+":\n" + str( progression ) + '\n' )

# MAPPING DE NOTAS A CODIGOS ==================================================
def create_mapping( dict_path ):
    """Crea y guarda un diccionario con las notas y sus codigos"""
    note_dict = {}
    for i in range( 12 ):
        note_dict[ m21.pitch.Pitch(i).name ] = i  
        
    with open( dict_path, 'w' ) as f:
        json.dump( note_dict, f )
        
def load_note_dict(dict_path):
    """Cargar un diccionario con las notas y sus codigos"""
    with open( dict_path, 'r' ) as f:
        note_dict = json.load( f )
    return note_dict

# CREAR DATASET ===============================================================

def create_dataset(progressions):
    """Crear el dataset de entrada y salida para el modelo"""    
    input = progressions
    target = progressions    
    return input, target

# MAIN ========================================================================
if __name__ == '__main__':    
    
    #create_mapping(DICTIONARY_PATH)
    note_dict=load_note_dict(DICTIONARY_PATH)    
    progressions = process_midi_files(MIDI_PATH, note_dict) 
    input, target = create_dataset(progressions)

    save_progressions_to_file(progressions,OUTPUT_PATH)
    
    print(f"Procesado {len(progressions)} progresiones armónicas.")    
    print(f"Input shape: {input.shape}, Target shape: {target.shape}")
        
    