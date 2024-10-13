import os
import data_loader
import model
import train
import utils

path = './data/'
image_dir = os.path.join(path, 'CameraRGB/')
mask_dir = os.path.join(path, 'CameraMask/')
buffer_size = 500
batch_size = 32
epochs = 50

image_list, mask_list = data_loader.load_image_paths(image_dir, mask_dir)

dataset = data_loader.prepare_dataset(image_list, mask_list, batch_size, buffer_size)

model = model.unet_model(input_size=(96, 128, 3), n_filters=32, n_classes=23)

history = train.compile_and_train(model, dataset, epochs, buffer_size, batch_size)

train.show_predictions(model, dataset, num=5)
