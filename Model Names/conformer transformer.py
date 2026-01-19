# ==================== Cell 9 — Conformer Model (UNCHANGED ARCHITECTURE) ====================
# This is the SAME model architecture you provided.
# Only use create_conformer_model(input_shape=(127, 100, 1)) later in training.

import tensorflow as tf  # tensorflow ops
from tensorflow import keras  # keras base
from tensorflow.keras import layers  # keras layers
import numpy as np  # numpy arrays

print("[STEP 9] Loading Conformer model code (unchanged)...")

# ==================== GLU Activation ====================
class GLU(layers.Layer):
    """Gated Linear Unit - splits input in half and applies gating"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        a, b = tf.split(x, 2, axis=-1)  # split channels into 2 halves
        return a * tf.nn.sigmoid(b)  # gate: a * sigmoid(b)

# ==================== Depthwise Conv1D ====================
class DepthwiseConv1D(layers.Layer):
    """
    Custom Depthwise 1D Convolution using groups.
    TF >= 2.4 required for Conv1D(groups=...).
    """
    def __init__(self, kernel_size, padding='same', **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size  # kernel size
        self.padding = padding  # padding mode

    def build(self, input_shape):
        self.channels = input_shape[-1]  # number of channels
        self.conv = layers.Conv1D(
            filters=self.channels,  # output channels = input channels
            kernel_size=self.kernel_size,  # kernel size
            padding=self.padding,  # same padding
            groups=self.channels,  # depthwise conv
            use_bias=False  # no bias
        )

    def call(self, x):
        return self.conv(x)  # apply depthwise conv

# ==================== Transformer-XL Relative Multi-Head Attention ====================
class TransformerXLMultiHeadAttention(layers.Layer):
    """
    Transformer-XL relative multi-head attention with:
    - relative embeddings
    - relative shift trick
    - content bias u and position bias v
    """
    def __init__(self, d_model, num_heads, dropout_rate=0.1, max_len=5000, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model  # embedding dim
        self.num_heads = num_heads  # num heads
        self.dropout_rate = dropout_rate  # dropout
        self.max_len = max_len  # max sequence length

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.depth = d_model // num_heads  # per-head dim

    def build(self, input_shape):
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)  # pre-norm

        # Q, K, V projections
        self.wq = layers.Dense(self.d_model, use_bias=False)
        self.wk = layers.Dense(self.d_model, use_bias=False)
        self.wv = layers.Dense(self.d_model, use_bias=False)

        # relative projection W_r
        self.w_r = layers.Dense(self.d_model, use_bias=False)

        # sinusoidal relative position embeddings (2*max_len+1, d_model)
        pos = np.arange(-self.max_len, self.max_len + 1, dtype=np.float32)[:, None]
        dim = np.arange(self.d_model, dtype=np.float32)[None, :]
        angle = pos / (10000 ** (2 * (dim // 2) / self.d_model))
        pe = np.where(dim % 2 == 0, np.sin(angle), np.cos(angle)).astype(np.float32)

        self.rel_pos_emb = self.add_weight(
            name='rel_pos_emb',
            shape=(2 * self.max_len + 1, self.d_model),
            initializer=keras.initializers.Constant(pe),
            trainable=False
        )

        # biases u and v (per head)
        self.u_bias = self.add_weight(
            name='u_bias',
            shape=(self.num_heads, self.depth),
            initializer='zeros',
            trainable=True
        )
        self.v_bias = self.add_weight(
            name='v_bias',
            shape=(self.num_heads, self.depth),
            initializer='zeros',
            trainable=True
        )

        self.dense_out = layers.Dense(self.d_model)  # output projection
        self.dropout_attn = layers.Dropout(self.dropout_rate)  # dropout on attn weights
        self.dropout_out = layers.Dropout(self.dropout_rate)  # dropout on output

    def rel_shift(self, x):
        """
        Relative shift trick.
        Input:  (B, H, L, 2L-1)
        Output: (B, H, L, L)
        """
        batch_size = tf.shape(x)[0]
        heads = tf.shape(x)[1]
        seq_len = tf.shape(x)[2]

        x_padded = tf.pad(x, [[0, 0], [0, 0], [0, 0], [1, 0]])  # pad left
        x_padded = tf.reshape(x_padded, [batch_size, heads, 2 * seq_len, seq_len])  # reshape
        x_shifted = x_padded[:, :, 1:, :]  # drop first row
        x_shifted = tf.reshape(x_shifted, [batch_size, heads, seq_len, 2 * seq_len - 1])  # reshape back
        return x_shifted[:, :, :, :seq_len]  # keep first L columns

    def call(self, x, mask=None, training=False):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        # ensure seq_len <= max_len
        tf.debugging.assert_less_equal(seq_len, self.max_len, message="seq_len too large for max_len")

        x_norm = self.layer_norm(x)  # pre-norm

        q = self.wq(x_norm)  # (B, L, D)
        k = self.wk(x_norm)  # (B, L, D)
        v = self.wv(x_norm)  # (B, L, D)

        # reshape to (B, L, H, d)
        q = tf.reshape(q, [batch_size, seq_len, self.num_heads, self.depth])
        k = tf.reshape(k, [batch_size, seq_len, self.num_heads, self.depth])
        v = tf.reshape(v, [batch_size, seq_len, self.num_heads, self.depth])

        # transpose to (B, H, L, d)
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])

        # relative indices for L -> (2L-1)
        r_indices = tf.range(2 * seq_len - 1, dtype=tf.int32)
        r_indices_centered = r_indices - (seq_len - 1) + self.max_len
        r_emb = tf.gather(self.rel_pos_emb, r_indices_centered)  # (2L-1, D)

        # project relative embedding
        r = self.w_r(r_emb)  # (2L-1, D)
        r = tf.reshape(r, [2 * seq_len - 1, self.num_heads, self.depth])  # (2L-1, H, d)
        r = tf.transpose(r, [1, 0, 2])  # (H, 2L-1, d)

        # AC = (Q+u)K^T
        q_with_u = q + self.u_bias[None, :, None, :]
        AC = tf.matmul(q_with_u, k, transpose_b=True)  # (B, H, L, L)

        # BD = (Q+v)R^T then rel_shift
        q_with_v = q + self.v_bias[None, :, None, :]
        BD = tf.einsum('bhld,hrd->bhlr', q_with_v, r)  # (B, H, L, 2L-1)
        BD_shifted = self.rel_shift(BD)  # (B, H, L, L)

        attn_score = AC + BD_shifted  # combine
        attn_score = attn_score / tf.math.sqrt(tf.cast(self.depth, tf.float32))  # scale

        # optional mask
        if mask is not None:
            mask = tf.cast(mask, attn_score.dtype)
            attn_score += (1.0 - mask) * -1e9

        attn_weights = tf.nn.softmax(attn_score, axis=-1)  # softmax
        attn_weights = self.dropout_attn(attn_weights, training=training)  # dropout

        attn_output = tf.matmul(attn_weights, v)  # (B, H, L, d)

        # concat heads: (B, L, D)
        attn_output = tf.transpose(attn_output, [0, 2, 1, 3])
        attn_output = tf.reshape(attn_output, [batch_size, seq_len, self.d_model])

        output = self.dense_out(attn_output)  # output proj
        output = self.dropout_out(output, training=training)  # dropout
        return output

# ==================== Feed Forward Module ====================
class FeedForwardModule(layers.Layer):
    """FFN with 4x expansion and swish"""
    def __init__(self, d_model, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)  # pre-norm
        self.dense1 = layers.Dense(self.d_model * 4)  # expand 4x
        self.swish = layers.Activation('swish')  # swish
        self.dropout1 = layers.Dropout(self.dropout_rate)  # dropout
        self.dense2 = layers.Dense(self.d_model)  # project back
        self.dropout2 = layers.Dropout(self.dropout_rate)  # dropout

    def call(self, x, training=False):
        x_norm = self.layer_norm(x)
        x = self.dense1(x_norm)
        x = self.swish(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return x

# ==================== Convolution Module ====================
class ConvolutionModule(layers.Layer):
    """Conv module with GLU + depthwise conv"""
    def __init__(self, d_model, kernel_size=32, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)  # pre-norm
        self.pointwise_conv1 = layers.Conv1D(self.d_model * 2, kernel_size=1)  # expand
        self.glu = GLU()  # GLU gate
        self.depthwise_conv = DepthwiseConv1D(kernel_size=self.kernel_size, padding='same')  # depthwise
        self.batch_norm = layers.BatchNormalization()  # batch norm
        self.swish = layers.Activation('swish')  # swish
        self.pointwise_conv2 = layers.Conv1D(self.d_model, kernel_size=1)  # project back
        self.dropout = layers.Dropout(self.dropout_rate)  # dropout

    def call(self, x, training=False):
        x_norm = self.layer_norm(x)
        x = self.pointwise_conv1(x_norm)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x, training=training)
        x = self.swish(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x, training=training)
        return x

# ==================== Conformer Block ====================
class ConformerBlock(layers.Layer):
    """FFN(0.5) -> MHSA -> Conv -> FFN(0.5) -> LN"""
    def __init__(self, d_model, num_heads, kernel_size=32, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.ffn1 = FeedForwardModule(d_model, dropout_rate)
        self.mhsa = TransformerXLMultiHeadAttention(d_model, num_heads, dropout_rate)
        self.conv = ConvolutionModule(d_model, kernel_size, dropout_rate)
        self.ffn2 = FeedForwardModule(d_model, dropout_rate)
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, mask=None, training=False):
        x = x + 0.5 * self.ffn1(x, training=training)
        x = x + self.mhsa(x, mask=mask, training=training)
        x = x + self.conv(x, training=training)
        x = x + 0.5 * self.ffn2(x, training=training)
        return self.layer_norm(x)

# ==================== Conformer Encoder (Vision Adaptation) ====================
class ConformerEncoder(keras.Model):
    """Vision-style Conformer encoder for binary classification"""
    def __init__(self, input_shape_tuple, num_blocks=4, d_model=128, num_heads=4, kernel_size=32, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.input_shape_tuple = input_shape_tuple
        self.num_blocks = num_blocks
        self.d_model = d_model

        # Conv subsampling (same architecture)
        self.conv_subsample = keras.Sequential([
            layers.Conv2D(d_model // 2, kernel_size=3, strides=2, padding='same'),
            layers.ReLU(),
            layers.Conv2D(d_model, kernel_size=3, strides=2, padding='same'),
            layers.ReLU(),
        ])

        self.conformer_blocks = [
            ConformerBlock(d_model, num_heads, kernel_size, dropout_rate)
            for _ in range(num_blocks)
        ]

        self.global_pool = layers.GlobalAveragePooling1D()
        self.final_dropout = layers.Dropout(dropout_rate)
        self.classifier = layers.Dense(1, activation='sigmoid')

    def call(self, x, training=False):
        x = self.conv_subsample(x, training=training)

        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]

        x = tf.reshape(x, [batch_size, height * width, self.d_model])

        for block in self.conformer_blocks:
            x = block(x, mask=None, training=training)

        x = self.global_pool(x)
        x = self.final_dropout(x, training=training)
        return self.classifier(x)

# ==================== Model Creation ====================
def create_conformer_model(input_shape=(127, 100, 1)):
    """Create and compile Conformer model (same architecture)"""
    model = ConformerEncoder(
        input_shape_tuple=input_shape,
        num_blocks=4,
        d_model=128,
        num_heads=4,
        kernel_size=32,
        dropout_rate=0.1
    )

    dummy = tf.zeros((1,) + input_shape)  # dummy input
    _ = model(dummy, training=False)  # build model

    model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),  # optimizer
    loss='binary_crossentropy',  # loss
    metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5)]  # metric
)


    return model

print("[STEP 9] ✅ Conformer model code ready.")
print("[STEP 9] Example: model = create_conformer_model(input_shape=(127, 100, 1))")
