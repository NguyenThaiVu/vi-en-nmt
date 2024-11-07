import numpy as np
import tensorflow as tf


QUANTIZED_TYPE = tf.int8 # tf.int8, tf.uint8, tf.float16
QUANTIZED_TECHNIQUE = "asymmetric"  # symmetric or asymmetric


def positional_encoding(length, depth):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

  return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
    self.pos_encoding = positional_encoding(length=2048, depth=d_model)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    length = tf.shape(x)[1]
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x



class Custom_Quantization_MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, quantized_type=QUANTIZED_TYPE, quantized_technique=QUANTIZED_TECHNIQUE, dropout=None, **kwargs):
        super(Custom_Quantization_MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        
        self.quantized_type = quantized_type
        self.quantized_technique = quantized_technique  # symmetric or asymmetric
        self.dropout = dropout


    def build(self, input_shape):
        # Create the weight and bias variables
        self.wq = self.add_weight(shape=(input_shape[-1], self.key_dim, self.num_heads), initializer='random_normal', trainable=True)
        self.bq = self.add_weight(shape=(self.key_dim, self.num_heads), initializer='zeros', trainable=True)
        self.wq_quantized = self.add_weight(shape=(input_shape[-1], self.key_dim, self.num_heads), dtype=self.quantized_type, initializer='zeros', trainable=False)
        self.quantized_scale_q = self.add_weight(shape=(), initializer='ones', trainable=False)
        self.zero_point_q = self.add_weight(shape=(), initializer='ones', trainable=False)

        self.wk = self.add_weight(shape=(input_shape[-1], self.key_dim, self.num_heads), initializer='random_normal', trainable=True)
        self.bk = self.add_weight(shape=(self.key_dim, self.num_heads), initializer='zeros', trainable=True)
        self.wk_quantized = self.add_weight(shape=(input_shape[-1], self.key_dim, self.num_heads), dtype=self.quantized_type, initializer='zeros', trainable=False)
        self.quantized_scale_k = self.add_weight(shape=(), initializer='ones', trainable=False)
        self.zero_point_k = self.add_weight(shape=(), initializer='ones', trainable=False)

        self.wv = self.add_weight(shape=(input_shape[-1], self.key_dim, self.num_heads), initializer='random_normal', trainable=True)
        self.bv = self.add_weight(shape=(self.key_dim, self.num_heads), initializer='zeros', trainable=True)
        self.wv_quantized = self.add_weight(shape=(input_shape[-1], self.key_dim, self.num_heads), dtype=self.quantized_type, initializer='zeros', trainable=False)
        self.quantized_scale_v = self.add_weight(shape=(), initializer='ones', trainable=False)
        self.zero_point_v = self.add_weight(shape=(), initializer='ones', trainable=False)

        self.wo = self.add_weight(shape=(self.key_dim * self.num_heads, input_shape[-1]), initializer='random_normal', trainable=True)
        self.bo = self.add_weight(shape=(input_shape[-1]), initializer='zeros', trainable=True)
        self.wo_quantized = self.add_weight(shape=(self.key_dim * self.num_heads, input_shape[-1]), dtype=self.quantized_type, initializer='zeros', trainable=False)
        self.quantized_scale_o = self.add_weight(shape=(), initializer='ones', trainable=False)
        self.zero_point_o = self.add_weight(shape=(), initializer='ones', trainable=False)

        super(Custom_Quantization_MultiHeadAttention, self).build(input_shape)  # Be sure to call this at the end


    def get_symmetric_quantize_matrix(self, x, target_dtype):
        # Define the range for target type
        qmin = tf.constant(target_dtype.min, dtype=tf.float32)
        qmax = tf.constant(target_dtype.max, dtype=tf.float32)

        # Compute the scale factor
        max_abs_val = tf.reduce_max(tf.abs(x))
        scale = max_abs_val / qmax
        zero_point = 0.0

        # Quantization and Clip to ensure quantized values are within the range
        quantized = tf.round(x / scale)  # Perform quantization
        quantized = tf.clip_by_value(quantized, qmin, qmax) 
        
        return tf.cast(quantized, target_dtype), scale, zero_point


    def get_asymmetric_quantize_matrix(self, x, target_dtype):
        # Define the range for the target type
        qmin = tf.constant(target_dtype.min, dtype=tf.float32)
        qmax = tf.constant(target_dtype.max, dtype=tf.float32)
        
        # Compute the min and max for the actual tensor
        x_min = tf.reduce_min(x)
        x_max = tf.reduce_max(x)

        # Scale and zero point calculations
        scale = (x_max - x_min) / (qmax - qmin)
        zero_point = tf.cast(qmin - x_min / scale, tf.float32)

        # Quantization and Clip to ensure quantized values are within the range
        quantized = tf.round((x - x_min) / scale + qmin)
        quantized = tf.clip_by_value(quantized, qmin, qmax)

        return tf.cast(quantized, target_dtype), scale, zero_point


    def post_training_quantization(self):
        """Perform post training quantization after FINISH training"""

        # Query matrix
        if self.quantized_technique == 'symmetric':
            wq_quantized, quantized_scale_q, zero_point = self.get_symmetric_quantize_matrix(self.wq, target_dtype=self.quantized_type)
        elif self.quantized_technique == 'asymmetric':
            wq_quantized, quantized_scale_q, zero_point = self.get_asymmetric_quantize_matrix(self.wq, target_dtype=self.quantized_type)

        self.wq_quantized.assign(wq_quantized)
        self.quantized_scale_q.assign(quantized_scale_q)
        self.zero_point_q.assign(zero_point)

        # Key matrix
        if self.quantized_technique == 'symmetric':
            wk_quantized, quantized_scale_k, zero_point = self.get_symmetric_quantize_matrix(self.wk, target_dtype=self.quantized_type)
        elif self.quantized_technique == 'asymmetric':
            wk_quantized, quantized_scale_k, zero_point = self.get_asymmetric_quantize_matrix(self.wk, target_dtype=self.quantized_type)
    
        self.wk_quantized.assign(wk_quantized)
        self.quantized_scale_k.assign(quantized_scale_k)
        self.zero_point_k.assign(zero_point)

        # Value matrix
        if self.quantized_technique == 'symmetric':
            wv_quantized, quantized_scale_v, zero_point = self.get_symmetric_quantize_matrix(self.wv, target_dtype=self.quantized_type)
        elif self.quantized_technique == 'asymmetric':
            wv_quantized, quantized_scale_v, zero_point = self.get_asymmetric_quantize_matrix(self.wv, target_dtype=self.quantized_type)

        self.wv_quantized.assign(wv_quantized)
        self.quantized_scale_v.assign(quantized_scale_v)
        self.zero_point_v.assign(zero_point)

        # W_o matrix
        if self.quantized_technique == 'symmetric':
            wo_quantized, quantized_scale_o, zero_point = self.get_symmetric_quantize_matrix(self.wo, target_dtype=self.quantized_type)
        elif self.quantized_technique == 'asymmetric':
            wo_quantized, quantized_scale_o, zero_point = self.get_asymmetric_quantize_matrix(self.wo, target_dtype=self.quantized_type)

        self.wo_quantized.assign(wo_quantized)
        self.quantized_scale_o.assign(quantized_scale_o)
        self.zero_point_o.assign(zero_point)

        del self.wq, self.wk, self.wv, self.wo


    def call(self, query, key, value):

        if hasattr(self, 'wq'):
            q = tf.einsum('bij,jkh->bikh', query, self.wq) + self.bq
            k = tf.einsum('bij,jkh->bikh', key, self.wk) + self.bk
            v = tf.einsum('bij,jkh->bikh', value, self.wv) + self.bv
        else:
            # Scale back the weight (de-quantized)
            wq_dequantized = tf.cast(self.wq_quantized, tf.float32) - self.zero_point_q
            wq_dequantized = tf.multiply(wq_dequantized, self.quantized_scale_q)
            q = tf.einsum('bij,jkh->bikh', query, wq_dequantized) + self.bq

            wk_dequantized = tf.cast(self.wk_quantized, tf.float32) - self.zero_point_k
            wk_dequantized = tf.multiply(wk_dequantized, self.quantized_scale_k)
            k = tf.einsum('bij,jkh->bikh', key, wk_dequantized) + self.bk

            wv_dequantized = tf.cast(self.wv_quantized, tf.float32) - self.zero_point_v
            wv_dequantized = tf.multiply(wv_dequantized, self.quantized_scale_v)
            v = tf.einsum('bij,jkh->bikh', value, wv_dequantized) + self.bv
            
        # Scaled dot-product attention
        k_t = tf.transpose(k, perm=[0, 2, 1, 3])
        matmul_qk = tf.einsum('bijh,bjkh->bikh', q, k_t)
        dk = tf.cast(self.key_dim, tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-2)

        attention_output = tf.einsum('bijh,bjkh->bikh', attention_weights, v)

        # Final linear layer
        batch_size = tf.shape(q)[0]
        input_length = tf.shape(q)[1]
        # attention_output = tf.reshape(attention_output, [batch_size, attention_output.shape[1], -1])
        attention_output = tf.reshape(attention_output, [batch_size, input_length, -1])

        if hasattr(self, 'wo'):
            output = tf.matmul(attention_output, self.wo) + self.bo
        else:
            wo_dequantized = tf.cast(self.wo_quantized, tf.float32) - self.zero_point_o
            wo_dequantized = tf.multiply(wo_dequantized, self.quantized_scale_o)
            output = tf.matmul(attention_output, wo_dequantized) + self.bo

        return output


class CustomDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, quantized_type=tf.int8, quantized_technique='asymmetric'):
        super(CustomDense, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.quantized_type = quantized_type
        self.quantized_technique = quantized_technique  # symmetric or asymmetric

    def get_symmetric_quantize_matrix(self, x, target_dtype):
        # Define the range for target type
        qmin = tf.constant(target_dtype.min, dtype=tf.float32)
        qmax = tf.constant(target_dtype.max, dtype=tf.float32)

        # Compute the scale factor
        max_abs_val = tf.reduce_max(tf.abs(x))
        scale = max_abs_val / qmax
        zero_point = 0.0

        # Quantization and Clip to ensure quantized values are within the range
        quantized = tf.round(x / scale)  # Perform quantization
        quantized = tf.clip_by_value(quantized, qmin, qmax) 
        
        return tf.cast(quantized, target_dtype), scale, zero_point


    def get_asymmetric_quantize_matrix(self, x, target_dtype):
        # Define the range for the target type
        qmin = tf.constant(target_dtype.min, dtype=tf.float32)
        qmax = tf.constant(target_dtype.max, dtype=tf.float32)
        
        # Compute the min and max for the actual tensor
        x_min = tf.reduce_min(x)
        x_max = tf.reduce_max(x)

        # Scale and zero point calculations
        scale = (x_max - x_min) / (qmax - qmin)
        zero_point = tf.cast(qmin - x_min / scale, tf.float32)

        # Quantization and Clip to ensure quantized values are within the range
        quantized = tf.round((x - x_min) / scale + qmin)
        quantized = tf.clip_by_value(quantized, qmin, qmax)

        return tf.cast(quantized, target_dtype), scale, zero_point


    def build(self, input_shape):
        # Initialize weights and bias
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer='random_normal', trainable=True)

        self.w_quantized = self.add_weight(shape=(input_shape[-1], self.units), dtype=self.quantized_type, initializer='zeros', trainable=False)
        self.quantized_scale = self.add_weight(shape=(), initializer='ones', trainable=False)
        self.zero_point = self.add_weight(shape=(), initializer='ones', trainable=False)

        super(CustomDense, self).build(input_shape)  # Be sure to call this at the end


    def post_training_quantization(self):
        """Perform post training quantization after FINISH training"""

        # Query matrix
        if self.quantized_technique == 'symmetric':
            w_quantized, quantized_scale, zero_point = self.get_symmetric_quantize_matrix(self.w, target_dtype=self.quantized_type)
        elif self.quantized_technique == 'asymmetric':
            w_quantized, quantized_scale, zero_point = self.get_asymmetric_quantize_matrix(self.w, target_dtype=self.quantized_type)

        self.w_quantized.assign(w_quantized)
        self.quantized_scale.assign(quantized_scale)
        self.zero_point.assign(zero_point)

        del self.w

    def call(self, inputs):
        
        # Linear transformation
        if hasattr(self, 'w'):
            z = tf.matmul(inputs, self.w) + self.b  
        else:
            w_dequantized = tf.cast(self.w_quantized, tf.float32) - self.zero_point
            w_dequantized = tf.multiply(w_dequantized, self.quantized_scale)
            z = tf.matmul(inputs, w_dequantized) + self.b

        if self.activation:
            z = self.activation(z)  
        return z  



class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-7)
    self.add = tf.keras.layers.Add()


class BaseAttention_Quant(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = Custom_Quantization_MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-7)
    self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention_Quant):
  def call(self, x, context):
    attn_output = self.mha(
        query=x,
        key=context,
        value=context)

    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x
  

class GlobalSelfAttention(BaseAttention_Quant):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x
  

class CausalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x,
        use_causal_mask = True)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x
  

class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      CustomDense(dff, activation='relu'),
      CustomDense(d_model),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-7)

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x) 
    return x
  

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x):
    x = self.self_attention(x)
    x = self.ffn(x)
    return x
  

class Encoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads,
               dff, vocab_size, dropout_rate=0.1):
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)

    self.enc_layers = [
        EncoderLayer(d_model=d_model,
                     num_heads=num_heads,
                     dff=dff,
                     dropout_rate=dropout_rate)
        for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x):
    # `x` is token-IDs shape: (batch, seq_len)
    x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

    # Add dropout.
    x = self.dropout(x)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x)

    return x  # Shape `(batch_size, seq_len, d_model)`.
  

class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self,
               *,
               d_model,
               num_heads,
               dff,
               dropout_rate=0.1):
    super(DecoderLayer, self).__init__()

    self.causal_self_attention = CausalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.cross_attention = CrossAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x, context):
    x = self.causal_self_attention(x=x)
    x = self.cross_attention(x=x, context=context)

    x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
    return x
  

class Decoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size,
               dropout_rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.dec_layers = [
        DecoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate)
        for _ in range(num_layers)]


  def call(self, x, context):
    # `x` is token-IDs shape (batch, target_seq_len)
    x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

    x = self.dropout(x)

    for i in range(self.num_layers):
      x  = self.dec_layers[i](x, context)

    # The shape of x is (batch_size, target_seq_len, d_model).
    return x
  

class Transformer(tf.keras.Model):
  def __init__(self, *, num_layers, d_model, num_heads, dff,
               input_vocab_size, target_vocab_size, dropout_rate=0.1):
    super().__init__()
    self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=input_vocab_size,
                           dropout_rate=dropout_rate)

    self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=target_vocab_size,
                           dropout_rate=dropout_rate)

    self.final_layer = CustomDense(target_vocab_size)

  def call(self, inputs):
    # To use a Keras model with `.fit` you must pass all your inputs in the first argument.
    context, x  = inputs

    context = self.encoder(context)  # (batch_size, context_len, d_model)

    x = self.decoder(x, context)  # (batch_size, target_len, d_model)

    # Final linear layer output.
    logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

    try:
      # Drop the keras mask, so it doesn't scale the losses/metrics. b/250038731
      del logits._keras_mask
    except AttributeError:
      pass

    # Return the final output and the attention weights.
    return logits