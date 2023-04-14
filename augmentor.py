import os
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def augment_images(datagen, parent_dir, save_dir, prefix, num_augmented=5):
    # Create the output directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    for filename in os.listdir(parent_dir):
        file_path = os.path.join(parent_dir, filename)
        fname = filename.split('.')[0]
        
        # Load the image as a numpy array
        img = load_img(file_path)
        x = img_to_array(img) 
        x = x.reshape((1,) + x.shape) 
        
        # Generate and save augmented images
        i = 0
        for batch in datagen.flow(x, batch_size=32, save_to_dir=save_dir, 
                                  save_prefix=prefix + fname, save_format='png'):
            i += 1
            if i >= num_augmented:
                break

def main():
    # Define the data generator with the desired augmentation parameters
    datagen = ImageDataGenerator(shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

    # Define the directories and prefixes for input and output images
    input_dir = "Dataset/train/"
    output_dir = "Dataset/aug/"
    prefixes = ['Covid-', 'Normal-', 'Pneumonia-']

    # Loop over the input directories and augment each set of images
    for prefix in prefixes:
        input_path = os.path.join(input_dir, prefix[:-1])
        output_path = os.path.join(output_dir, prefix[:-1])
        augment_images(datagen, input_path, output_path, prefix)

if __name__ == '__main__':
    main()
