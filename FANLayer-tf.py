import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import constraints, activations, initializers, regularizers
from tensorflow.keras.constraints import NonNeg
from tensorflow.keras.constraints import Constraint

class FANLayer(tf.keras.layers.Layer):
    """
    FANLayer: The layer used in FAN (https://arxiv.org/abs/2410.02675).

    Args:
        input_dim (int): The number of input features.
        output_dim (int): The number of output features.
        p_ratio (float): The ratio of output dimensions used for cosine and sine parts (default: 0.25).
        activation (str or callable): The activation function to apply to the g component (default: 'gelu').
        use_p_bias (bool): If True, include bias in the linear transformations of the p component (default: True).
        gated (bool): If True, applies gating to the output.
        kernel_regularizer: Regularizer for kernel weights.
        bias_regularizer: Regularizer for bias weights.
    """
    
    def __init__(self, 
                 output_dim, 
                 p_ratio=0.25, 
                 activation='gelu', 
                 use_p_bias=True, 
                 gated=False, 
                 kernel_regularizer=None, 
                 bias_regularizer=None, 
                 **kwargs):
        super(FANLayer, self).__init__(**kwargs)
        
        assert 0 < p_ratio < 0.5, "p_ratio must be between 0 and 0.5"
        
        self.p_ratio = p_ratio
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        self.use_p_bias = use_p_bias
        self.gated = gated
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        
        # Compute output dimensions for p and g components
        self.p_output_dim = int(output_dim * self.p_ratio)
        self.g_output_dim = output_dim - 2 * self.p_output_dim  # Account for cosine and sine
        
        # Layers for linear transformations
        self.input_linear_p = layers.Dense(self.p_output_dim, 
                                    use_bias=self.use_p_bias, 
                                    kernel_regularizer=self.kernel_regularizer, 
                                    bias_regularizer=self.bias_regularizer)
        self.input_linear_g = layers.Dense(self.g_output_dim, 
                                    kernel_regularizer=self.kernel_regularizer, 
                                    bias_regularizer=self.bias_regularizer)
        
        if self.gated:
            self.gate = self.add_weight(name='gate', 
                                        shape=(1,), 
                                        initializer=initializers.RandomNormal(), 
                                        trainable=True, 
                                            regularizer=None, 
                                            constraint=NonNeg())

    def call(self, inputs):
        # Apply the linear transformation followed by the activation for the g component
        g = self.activation(self.input_linear_g(inputs))
        
        # Apply the linear transformation for the p component
        p = self.input_linear_p(inputs)
        
        if self.gated:
            gate = tf.sigmoid(self.gate)
            output = tf.concat([gate * tf.cos(p), gate * tf.sin(p), (1 - gate) * g], axis=-1)
        else:
            output = tf.concat([tf.cos(p), tf.sin(p), g], axis=-1)
        
        return output

    def get_config(self):
        config = super(FANLayer, self).get_config()
        config.update({
            "output_dim": self.output_dim,
            "p_ratio": self.p_ratio,
            "activation": activations.serialize(self.activation),
            "use_p_bias": self.use_p_bias,
            "gated": self.gated,
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer)
        })
        return config
