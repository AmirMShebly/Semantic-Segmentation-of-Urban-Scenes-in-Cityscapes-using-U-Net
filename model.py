import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate


def unet_model(input_size=(96, 128, 3), n_filters=32, n_classes=23):
    inputs = Input(input_size)

    # encoder
    c1 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    c1 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)
    
    c2 = Conv2D(n_filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p1)
    c2 = Conv2D(n_filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c2)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)

    c3 = Conv2D(n_filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p2)
    c3 = Conv2D(n_filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c3)
    p3 = MaxPooling2D(pool_size=(2, 2))(c3)

    c4 = Conv2D(n_filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p3)
    c4 = Conv2D(n_filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(0.3)(p4)

    # bottleneck
    c5 = Conv2D(n_filters * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(p4)
    c5 = Conv2D(n_filters * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c5)
    c5 = Dropout(0.3)(c5)

    # decoder
    u6 = Conv2DTranspose(n_filters * 8, 3, strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(n_filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(u6)
    c6 = Conv2D(n_filters * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c6)

    u7 = Conv2DTranspose(n_filters * 4, 3, strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(n_filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(u7)
    c7 = Conv2D(n_filters * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c7)

    u8 = Conv2DTranspose(n_filters * 2, 3, strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(n_filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(u8)
    c8 = Conv2D(n_filters * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c8)

    u9 = Conv2DTranspose(n_filters, 3, strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(u9)
    c9 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c9)

    
    outputs = Conv2D(n_classes, 1, activation='softmax', padding='same')(c9)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model