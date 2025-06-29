import tensorflow as tf
import numpy as np
import math

# TensorFlow 2.x compatibility layer
tf.compat.v1.disable_eager_execution()

class AdditiveAttention:
    def __init__(self, query_vector_dim, input_dim):
        self.query_vector_dim = query_vector_dim
        self.input_dim = input_dim
        
    def attention(self, inputs):
        # inputs: [batch_size, seq_len, input_dim]
        batch_size = tf.compat.v1.shape(inputs)[0]
        seq_len = tf.compat.v1.shape(inputs)[1]
        
        # Linear transformation
        W = tf.compat.v1.get_variable(
            "attention_W", 
            [self.input_dim, self.query_vector_dim],
            initializer=tf.compat.v1.keras.initializers.glorot_uniform()
        )
        
        # Query vector
        query = tf.compat.v1.get_variable(
            "attention_query",
            [self.query_vector_dim],
            initializer=tf.compat.v1.keras.initializers.glorot_uniform()
        )
        
        # Compute attention scores
        hidden = tf.compat.v1.nn.tanh(tf.compat.v1.tensordot(inputs, W, axes=1))
        scores = tf.compat.v1.tensordot(hidden, query, axes=1)
        
        # Apply softmax
        attention_weights = tf.compat.v1.nn.softmax(scores, axis=1)
        
        # Apply attention weights
        attended_output = tf.compat.v1.reduce_sum(
            tf.compat.v1.expand_dims(attention_weights, -1) * inputs, 
            axis=1
        )
        
        return attended_output

class MultiHeadSelfAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        
    def split_heads(self, x, batch_size):
        x = tf.compat.v1.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.compat.v1.transpose(x, perm=[0, 2, 1, 3])
    
    def attention(self, inputs):
        batch_size = tf.compat.v1.shape(inputs)[0]
        seq_len = tf.compat.v1.shape(inputs)[1]
        
        # Linear transformations for Q, K, V using tf.keras.layers.Dense
        with tf.compat.v1.variable_scope("multihead_attention", reuse=tf.compat.v1.AUTO_REUSE):
            # Replace tf.compat.v1.layers.dense with tf.keras.layers.Dense
            # Note: kernel_initializer is a direct Keras argument.
            # The name argument in Keras layers helps with variable scoping if needed, but Keras handles its own variable creation.
            self.query_projection_layer = tf.keras.layers.Dense(
                units=self.d_model,
                kernel_initializer=tf.compat.v1.keras.initializers.glorot_uniform(),
                name="query_projection"
            )
            self.key_projection_layer = tf.keras.layers.Dense(
                units=self.d_model,
                kernel_initializer=tf.compat.v1.keras.initializers.glorot_uniform(),
                name="key_projection"
            )
            self.value_projection_layer = tf.keras.layers.Dense(
                units=self.d_model,
                kernel_initializer=tf.compat.v1.keras.initializers.glorot_uniform(),
                name="value_projection"
            )
            self.output_projection_layer = tf.keras.layers.Dense(
                units=self.d_model,
                kernel_initializer=tf.compat.v1.keras.initializers.glorot_uniform(),
                name="output_projection"
            )

            Q = self.query_projection_layer(inputs)
            K = self.key_projection_layer(inputs)
            V = self.value_projection_layer(inputs)
        
        # Split heads
        Q = self.split_heads(Q, batch_size)  # (batch_size, num_heads, seq_len, depth)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        
        # Scaled dot-product attention
        attention_output = self.scaled_dot_product_attention(Q, K, V)
        
        # Concatenate heads
        attention_output = tf.compat.v1.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.compat.v1.reshape(attention_output, 
                                               (batch_size, seq_len, self.d_model))
        
        # Final linear projection
        # with tf.compat.v1.variable_scope("multihead_attention", reuse=tf.compat.v1.AUTO_REUSE): # Scope already handled by layer
        output = self.output_projection_layer(concat_attention)
        
        return output
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate the attention weights
        matmul_qk = tf.compat.v1.matmul(Q, K, transpose_b=True)
        
        # Scale the attention weights
        dk = tf.compat.v1.cast(tf.compat.v1.shape(K)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.compat.v1.math.sqrt(dk)
        
        # Apply mask if provided
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        # Apply softmax
        attention_weights = tf.compat.v1.nn.softmax(scaled_attention_logits, axis=-1)
        
        # Apply attention weights to values
        output = tf.compat.v1.matmul(attention_weights, V)
        
        return output 