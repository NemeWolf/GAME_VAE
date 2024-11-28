from models import cnn_vae_model, lstm_vae_model
import train
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist

# CNN PARAMETERS
MODEL_PATH = 'models\\CNN_VAE_model'

(_, x_test), (_, y_test) = mnist.load_data() # 60000 train, 10000 test            
# Normalize the pixel values
x_test = x_test.astype('float32') / 255. # 0-255 -> 0-1        
x_test = x_test.reshape(x_test.shape + (1 ,)) # (60000, 28, 28) -> (60000, 28, 28, 1)

TEST = x_test

y_test = y_test.astype('float32') / 255. # 0-255 -> 0-1        
y_test = y_test.reshape(y_test.shape + (1 ,)) # (60000, 28, 28) -> (60000, 28, 28, 1)

LABEL = y_test

class Evaluate():
    def __init__(self, test, label, model_path):
        self.vae = None        
        self.test_data = test
        self.label_data = label        
        self.model_path = model_path        
        self.reconstructured_data = None
        self.latent_representation = None        
        self.history = None        
        self.call()
        
    def call(self):
        self.load()
        self.plot_history()
        
    def load(self):
        """Carga los pesos y los parametros del modelo"""
        parameters_path = os.path.join(self.model_path , "parameters_vae.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)  
                           
        self.vae = cnn_vae_model.CNN_VAE(*parameters) # desempaquetamos los parametros y creamos un nuevo objeto autoencoder 
        
        weights_path = os.path.join(self.model_path , "weights_vae.weights.h5") 
        self.vae.model.load_weights(weights_path)
                
        history_path = os.path.join(self.model_path, 'training_history.pkl')
        with open(history_path, 'rb') as f:
            self.history = pickle.load(f)
    
    def plot_history(self):
        
            
        fig, axs = plt.subplots(2)

        # create accuracy subplot
        axs[0].plot(self.history.history['accuracy'], label='train accuracy')
        axs[0].plot(self.history.history['val_accuracy'], label='test accuracy')
        axs[0].set_ylabel('Accuracy')
        axs[0].legend(loc='lower right')
        axs[0].set_title('Accuracy eval')

        # create error subplot
        axs[1].plot(self.history.history['loss'], label='train error')
        axs[1].plot(self.history.history['val_loss'], label='test error')
        axs[1].set_ylabel('Error')
        axs[1].set_xlabel('Epoch')
        axs[1].legend(loc='upper right')
        axs[1].set_title('Error eval')

        plt.show()
        
    def reconstruct(self):
        self.latent_representation = self.vae.encoder.predict(self.sample_data) # codifica las imagenes en el espacio latente
        self.reconstructured_data = self.vae.decoder.predict(self.latent_representation) # reconstruye las imagenes
    

if __name__ == "__main__":    
    evalaute = Evaluate(TEST, LABEL, MODEL_PATH)
    
    
    