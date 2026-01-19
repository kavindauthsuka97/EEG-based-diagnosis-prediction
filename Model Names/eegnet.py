def get_lr(opt):  # helper
    """Safely read optimizer LR."""  # docstring
    try:
        return float(tf.keras.backend.get_value(opt.learning_rate))  # standard
    except Exception:
        try:
            return float(opt.learning_rate.numpy())  # alt
        except Exception:
            try:
                return float(opt.lr.numpy())  # legacy
            except Exception:
                return float(getattr(opt, "lr", 0.0))  # fallback

def create_eegnet(input_shape, dropout_rate=0.5, num_classes=1):  # EEGNet-ish
    """EEGNet (adapted to your pipeline)."""  # docstring
    n_electrodes = input_shape[0]  # channels
    inputs = Input(shape=input_shape)  # input

    x = Conv2D(16, (1, 64), padding='same', use_bias=False)(inputs)  # temporal conv
    x = BatchNormalization()(x)  # BN

    x = DepthwiseConv2D((n_electrodes, 1), depth_multiplier=2, padding='valid', use_bias=False)(x)  # spatial depthwise
    x = BatchNormalization()(x)  # BN
    x = Activation('elu')(x)  # ELU
    x = AveragePooling2D((1, 4))(x)  # pool
    x = Dropout(dropout_rate)(x)  # dropout

    x = SeparableConv2D(16, (1, 16), padding='same', use_bias=False)(x)  # separable conv
    x = BatchNormalization()(x)  # BN
    x = Activation('elu')(x)  # ELU
    x = AveragePooling2D((1, 8))(x)  # pool
    x = Dropout(dropout_rate)(x)  # dropout

    x = Flatten()(x)  # flatten
    x = Dense(64, activation='relu')(x)  # dense
    outputs = Dense(num_classes, activation='sigmoid')(x)  # sigmoid output

    return Model(inputs=inputs, outputs=outputs, name="EEGNet_simple")  # model

print("✅ Model function defined.")  # log
