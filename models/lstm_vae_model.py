from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Layer, RepeatVector, TimeDistributed, Reshape
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import os
import pickle
import tensorflow as tf

# LAYER DE PÉRDIDA VAE ==========================================================

class VAELossLayer(Layer):
    def __init__(self, reconstruction_loss_weight=1000, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)
        self.reconstruction_loss_weight = reconstruction_loss_weight

    def call(self, inputs):
        y_true, y_pred, mu, log_var = inputs

        # Pérdida de reconstrucción
        reconstruction_loss = tf.reduce_mean(tf.square(y_true - y_pred))

        # Pérdida KL divergente
        kl_loss = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=1)
        kl_loss = tf.reduce_mean(kl_loss)

        # Pérdida combinada
        combined_loss = self.reconstruction_loss_weight * reconstruction_loss + kl_loss
        self.add_loss(combined_loss)

        return y_pred  # Retorna y_pred para mantener la compatibilidad con Keras

# SAMPLE POINT FROM NORMAL DISTRIBUTION =========================================
class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# AUTOENCODER ===================================================================

class LSTM_VAE:

  def __init__(self, input_shape, latent_space_dim):
    super(LSTM_VAE, self).__init__()

  # INICIALIZAR LOS ATRIBUTOS DE LA CLASE AUTOENCODER ==========================
    self.input_shape = input_shape            
    self.latent_space_dim = latent_space_dim 
    self.reconstruction_loss_weights = 1000 
    
    self.encoder = None
    self.decoder = None
    self.model = None 
    
    self.sampling = Sampling()
    
    self._shape_before_bottleneck = None
    self._model_input = None

  # CONSTRUIR EL MODELO ==========================================================
    self._build()
                                  
  # CONSTURIMOS EL AUTOENCODER ===================================================
  def _build(self):
    """Construye todo el modelo autoencoder"""
    self._build_encoder()
    self._build_decoder()
    self._build_lstm_vae()
    
  # CONSTRUMOS AUTOENCODER ========================================================
  def _build_lstm_vae(self):
          
      model_input = self._model_input
      encoder_output, mu, log_var = self.encoder(model_input)
      decoder_output = self.decoder(encoder_output)

      vae_loss_layer = VAELossLayer(reconstruction_loss_weight=self.reconstruction_loss_weights)
      model_output= vae_loss_layer([model_input, decoder_output, mu, log_var ])

      self.model = Model(model_input, model_output, name="LSTM_VAE")   
    
  # CONSTRUIMOS ENCODER ============================================================
    
  def _build_encoder(self):
    """Construye el encoder del autoencoder"""
    
    encoder_inputs = Input(shape=self.input_shape, name="encoder_input")
    x = LSTM(64, return_sequences=True, name="encoder_lstm_1")(encoder_inputs)
    x = LSTM(64, return_sequences=False, name="encoder_lstm_2")(x)
    
    self._shape_before_bottleneck = K.int_shape(x)[1:] 
    
    x = Dense(128, activation='relu', name="encoder_dense")(x)
    
    mu = Dense(self.latent_space_dim, name="mu")(x)
    log_var = Dense(self.latent_space_dim, name="log_var")(x)
    
    encoder_outputs = Sampling()([mu, log_var])
    
    self._model_input = encoder_inputs  
    self.encoder = Model(encoder_inputs, [encoder_outputs, mu, log_var], name="encoder") 
  
  # CONSTRUIMOS DECODER============================================================
  
  def _build_decoder(self):
    """Construye el decoder del autoencoder"""

    decoder_inputs = Input(shape=(self.latent_space_dim,), name="decoder_input")
    
    x = Dense(128, activation='relu', name="decoder_dense")(decoder_inputs)
    x = Dense(np.prod(self._shape_before_bottleneck), activation='relu')(x)
    x = RepeatVector(self.input_shape[0])(x)
    x = LSTM(64, return_sequences=True, name="decoder_lstm_1")(x)
    x = LSTM(64, return_sequences=True, name="decoder_lstm_2")(x)
    decoder_outputs = TimeDistributed(Dense(self.input_shape[1], activation='softmax'), name="decoder_output")(x)
        
    self.decoder = Model(decoder_inputs, decoder_outputs, name="decoder")    
    self.decoder = Model(decoder_inputs, decoder_outputs, name="decoder")

# AUTOENCODER ===================================================================
if __name__ == "__main__":
  vae = LSTM_VAE(
      input_shape=(20, 13),
      latent_space_dim=2
  )
  
  vae.encoder.summary()
  vae.decoder.summary()
  vae.model.summary()