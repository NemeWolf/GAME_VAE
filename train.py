from models import cnn_vae_model, lstm_vae_model
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import os
import pickle
import numpy as np
from data.MIDI.preprocessing import process_midi_files, load_note_dict, DICTIONARY_PATH, MIDI_PATH

# CNN PARAMETERS
LEARNING_RATE_CNN = 0.0005
BATCH_SIZE_CNN = 32
EPOCHS_CNN = 150
SAVE_FOLDER_CNN = 'models\\CNN_VAE_model'

# LSTM PARAMETERS
LEARNING_RATE_LSTM = 0.0005
BATCH_SIZE_LSTM = 20
EPOCHS_LSTM = 100
SAVE_FOLDER_LSTM = 'models\\LSTM_VAE_model'

class Train():
    def __init__(self, model, train, test, epochs=10, batch_size=32, learning_rate=0.001, save_path="models"):
        self.vae = model
        self.train_data = train
        self.test_data = test
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.save_path = save_path
        self.history = None
        self.call()

    
    def call(self):
        self.summary()
        self.compile()
        self.train()
        self.save()
    
    def summary(self):        
        self.vae.encoder.summary()
        self.vae.decoder.summary()
        self.vae.model.summary()
        
    def compile(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Define métricas personalizadas
        def reconstruction_loss(y_true, y_pred):
            reconstruction_loss = tf.reduce_mean(tf.square(y_true - y_pred))
            return reconstruction_loss

        def kl_loss(y_true, y_pred):
            mu, log_var = self.vae.encoder(y_true)[1:]
            kl_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1)
            return tf.reduce_mean(kl_loss)

        # Compilamos con métricas personalizadas
        self.vae.model.compile(
            optimizer=optimizer,
            loss=None,  # La pérdida es calculada internamente en VAELossLayer
            metrics=[reconstruction_loss, kl_loss]
        )
    
    def train(self):
        if self.train_data is None and self.test_data is None:
            print("Loading MNIST dataset...")
            (self.train_data, _), (_, _) = mnist.load_data() # 60000 train, 10000 test            
            # Normalize the pixel values
            self.train_data = self.train_data[10000].astype('float32') / 255. # 0-255 -> 0-1        
            self.train_data = self.train_data.reshape(self.train_data.shape + (1 ,)) # (60000, 28, 28) -> (60000, 28, 28, 1)

        self.history = self.vae.model.fit(
            self.train_data,
            self.train_data,
            batch_size=self.batch_size,
            epochs=self.epochs,
            shuffle=True
        )

        
    # SALVAR EL MODELO ============================================================
    def save(self):        
        self._save_parameters() 
        self._save_weights()
        self._save_history()
        
    def _save_parameters(self):
        parameters = [self.vae.input_shape, self.vae.latent_space_dim]
        parameters_path = os.path.join(self.save_path, "parameters_vae.pkl")
        with open(parameters_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self):
        weights_path = os.path.join(self.save_path, "weights_vae.weights.h5")
        self.vae.model.save_weights(weights_path) 

    def _save_history(self):
        history_path = os.path.join(self.save_path, 'training_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(self.history, f)
        
if __name__ == '__main__': 
    
    # CNN dataset
    
    #CNN_VAE = cnn_vae_model.CNN_VAE(input_shape=(28, 28, 1), latent_space_dim=2)
    #Train_CNN_VAE = Train(CNN_VAE, 
                        #   train=None, 
                        #   test=None, 
                        #   epochs=EPOCHS_CNN, 
                        #   batch_size=BATCH_SIZE_CNN, 
                        #   learning_rate=LEARNING_RATE_CNN, 
                        #   save_path=SAVE_FOLDER_CNN)
    
    
    # LSTM dataset
    
    note_dict=load_note_dict(DICTIONARY_PATH)
    progressions = process_midi_files(MIDI_PATH, note_dict)
    
    LSTM_VAE = lstm_vae_model.LSTM_VAE(input_shape=(20,13), latent_space_dim=2)
    Train_CNN_VAE = Train(LSTM_VAE, 
                          train=np.expand_dims(progressions[100], axis=0), 
                          test=None, 
                          epochs=EPOCHS_LSTM, 
                          batch_size=BATCH_SIZE_LSTM, 
                          learning_rate=LEARNING_RATE_LSTM, 
                          save_path=SAVE_FOLDER_LSTM)
    
