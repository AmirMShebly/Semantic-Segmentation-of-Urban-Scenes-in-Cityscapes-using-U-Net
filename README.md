# Semantic Segmentation of Urban Scenes in Cityscapes using U-Net

Urban scenes are complex environments filled with diverse objects and structures, ranging from buildings and vehicles to pedestrians and trees. Understanding the
composition of these scenes is crucial for a wide range of applications, including autonomous driving, robotics, and urban planning.

Semantic segmentation is a computer vision task that aims to assign a label to each pixel in an image and plays a vital role in interpreting and understanding urban environments.

## Importance of Urban Scene Segmentation

• Autonomous Driving: Self-driving cars rely heavily on semantic segmentation to understand the road, identify obstacles, and navigate safely.

• Robotics: Robots need to understand their surroundings to perform tasks such as navigation, object manipulation, and human-robot interaction.

• Urban Planning: City planners can utilize semantic segmentation to analyze the distribution of buildings, green spaces, and traffic patterns, leading to better urban design and resource allocation.

## Cityscapes Dataset

This project leverages a portion of the Cityscapes dataset, a widely recognized benchmark for urban scene understanding tasks. Cityscapes provides a diverse collection of high-resolution images captured from street views, annotated with detailed pixel-level labels for various objects, including roads, buildings, vehicles, pedestrians, and more.

## U-Net: A Powerful Tool for Semantic Segmentation

The model used in this project is based on the U-Net architecture, a convolutional neural network specifically designed for image segmentation tasks.

It uses an encoder-decoder structure, allowing it to capture both local and global context within the image. The encoder downsamples the input image to extract features, while the decoder upsamples these features to generate a dense output segmentation map.

The network employs skip connections, which combine features from different levels of the encoder and decoder, enabling it to learn detailed and accurate object boundaries. 

U-Net's architecture is relatively efficient to train, making it practical for large datasets like Cityscapes.

## Results

![2](https://github.com/user-attachments/assets/cf452c90-e7cb-42f5-8c48-03003aea4241)
![3](https://github.com/user-attachments/assets/02ae07ad-7dd1-4a14-bc47-c8b0bf8ab760)
