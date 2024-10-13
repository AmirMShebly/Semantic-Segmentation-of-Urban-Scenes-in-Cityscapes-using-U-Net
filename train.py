import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy

def compile_and_train(model, dataset, epochs, buffer_size, batch_size):
    model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy, metrics=['accuracy'])
    
    history = model.fit(dataset, epochs=epochs)
    return history

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(model, dataset, num=1):
    for image, mask in dataset.take(num):
        pred_mask = model.predict(image)
        display([image[0], mask[0], create_mask(pred_mask)])
