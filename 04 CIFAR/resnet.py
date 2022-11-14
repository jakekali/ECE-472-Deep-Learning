import tensorflow as tf
import tensorflow_addons as tfa

batch_norm = True
gg = 32
aa = -1
DROPOUT = True

def identity_block(x, filters):
    x_skip = x
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding="same")(x)
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    else:
        x = tfa.layers.GroupNormalization(groups=gg, axis=aa)(x)
    if(DROPOUT): 
        x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(filters, (3, 3), padding="same")(x)
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    else:
        x = tfa.layers.GroupNormalization(groups=gg, axis=aa)(x)    
    if(DROPOUT): 
        x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Add()([x, x_skip])
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    else:
        x = tfa.layers.GroupNormalization(groups=gg, axis=aa)(x)     
    return x

def conv_block(x, filters):
    x_skip = x
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding="same", kernel_regularizer='l2')(x)
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    else:
        x = tfa.layers.GroupNormalization(groups=gg, axis=aa)(x)  

    if(DROPOUT): 
        x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(filters, (3, 3), padding="same",kernel_regularizer='l2')(x)
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    else:
        x = tfa.layers.GroupNormalization(groups=gg, axis=aa)(x)   

    if(DROPOUT): 
        x = tf.keras.layers.Dropout(0.1)(x)


    x_skip = tf.keras.layers.Conv2D(filters, (1, 1), padding="same", kernel_regularizer='l2')(x_skip)
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    else:
        x = tfa.layers.GroupNormalization(groups=gg, axis=aa)(x)      
    if(DROPOUT): 
        x = tf.keras.layers.Dropout(0.1)(x)


    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation("relu")(x)
    return x

def resnet_builder(output_dim=10, block_layers=[2,2,2,2], filters=[32,64,128,256]):
    x_IN = tf.keras.layers.Input(shape=(32, 32, 3))
    x = tf.cast(x_IN, tf.float32)
    x = tf.keras.applications.resnet50.preprocess_input(x_IN)
    x = tf.keras.layers.Conv2D(64, (3, 3))(x)
    if(DROPOUT): 
        x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Activation("relu")(x)
  
    for i in range(len(block_layers)):
        if i == 0: 
            for _ in range(block_layers[i]):
                x = identity_block(x, filters[i])
        else:
            x = conv_block(x, filters[i])
            for _ in range(block_layers[i]-1):
                x = identity_block(x, filters[i])

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    if(DROPOUT): 
        x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Dense(output_dim, activation="softmax")(x)
    model = tf.keras.models.Model(inputs = x_IN, outputs = x, name = "ResNet34")
    return model

dropout = 0.20