# evaluate_sr.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from skimage import io
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import logging
from tensorflow.keras.layers import (
    Input, Conv2D, UpSampling2D, MaxPooling2D, Add, Multiply, Activation,
    GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Reshape,
    Lambda, Concatenate, BatchNormalization
)
import keras

# === Configuration Variables ===

# Paths to your test dataset directories
hr_test_dir = '/mimer/NOBACKUP/groups/geodl/DeepRockSR-2D/shuffled2D/shuffled2D_test_HR'
lr_test_dir = '/mimer/NOBACKUP/groups/geodl/DeepRockSR-2D/shuffled2D/shuffled2D_test_LR_default_X2'

# Path to your trained model weights
model_weights_path = 'models/dual_branch_unet_with_attention_final_1.keras'  # Update as needed

# Parameters
target_hr_size = (500, 500)    # High-Resolution image size
target_lr_size = (250, 250)    # Low-Resolution image size
batch_size = 4                 # Batch size for evaluation
num_samples_to_visualize = 5   # Number of sample predictions to visualize
save_predictions_dir = 'plots' # Directory to save prediction plots

# === Set Up Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Define Custom Functions and Blocks ===

def channel_attention(input_feature, ratio=8):
    channel = K.int_shape(input_feature)[-1]
    shared_dense_one = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True)
    shared_dense_two = Dense(channel, kernel_initializer='he_normal', use_bias=True)

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_dense_one(avg_pool)
    avg_pool = shared_dense_two(avg_pool)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_dense_one(max_pool)
    max_pool = shared_dense_two(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    return Multiply()([input_feature, cbam_feature])

def spatial_attention(input_feature):
    """
    Spatial Attention Module with explicit output_shape for Lambda layers.
    """
    # Average and Max Pooling along the channel axis using Lambda layers
    avg_pool = Lambda(
        lambda x: K.mean(x, axis=-1, keepdims=True),
        output_shape=lambda s: (s[0], s[1], s[2], 1)
    )(input_feature)

    max_pool = Lambda(
        lambda x: K.max(x, axis=-1, keepdims=True),
        output_shape=lambda s: (s[0], s[1], s[2], 1)
    )(input_feature)

    # Concatenate along the channel axis
    concat = Concatenate(axis=-1)([avg_pool, max_pool])

    # Convolution and activation
    cbam_feature = Conv2D(
        1, kernel_size=7, strides=1, padding='same',
        activation='sigmoid', kernel_initializer='he_normal',
        use_bias=False
    )(concat)

    # Apply attention
    output_feature = Multiply()([input_feature, cbam_feature])
    return output_feature

def cbam_block(input_feature, ratio=8):
    x = channel_attention(input_feature, ratio)
    return spatial_attention(x)

# Define the dual-branch U-Net model architecture
def texture_branch(input_layer):
    """
    Texture branch of the dual-branch U-Net.
    """
    # Encoding
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = cbam_block(conv1)

    # Upsampling
    up1 = UpSampling2D((2, 2))(conv1)
    up_conv1 = Conv2D(32, 3, activation='relu', padding='same')(up1)
    up_conv1 = BatchNormalization()(up_conv1)
    up_conv1 = cbam_block(up_conv1)

    # Output layer
    output = Conv2D(1, 3, activation='sigmoid', padding='same')(up_conv1)
    return output

def structure_branch(input_layer):
    """
    Structure branch of the dual-branch U-Net.
    """
    # Encoding
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)

    # Downsampling
    pool1 = MaxPooling2D((2, 2))(conv1)  # (125, 125, 64)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)

    # Upsampling to original size
    up1 = UpSampling2D((2, 2))(conv2)    # (250, 250, 128)
    up_conv1 = Conv2D(64, 3, activation='relu', padding='same')(up1)
    up_conv1 = BatchNormalization()(up_conv1)

    # Skip connection
    concat1 = Add()([conv1, up_conv1])   # (250, 250, 64)

    # Additional Conv Layers
    conv3 = Conv2D(64, 3, activation='relu', padding='same')(concat1)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(64, 3, activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)

    up2 = UpSampling2D((2, 2))(conv3)    # (500, 500, 64)
    conv4 = Conv2D(32, 3, activation='relu', padding='same')(up2)
    conv4 = BatchNormalization()(conv4)

    # Output layer
    output = Conv2D(1, 3, activation='sigmoid', padding='same')(conv4)
    return output

def dual_branch_unet(input_shape):
    """
    Dual-branch U-Net model combining texture and structure branches.
    """
    inputs = Input(input_shape)

    # Texture branch
    texture_output = texture_branch(inputs)

    # Structure branch
    structure_output = structure_branch(inputs)

    # Fusion
    fused = Add()([texture_output, structure_output])
    fused = Conv2D(1, 1, activation='sigmoid', padding='same')(fused)

    model = tf.keras.models.Model(inputs=inputs, outputs=fused)
    return model

# Custom loss function and metrics
def combined_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    return mse + mae

def psnr_metric(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def ssim_metric(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)

# === Data Loading and Preprocessing Functions ===

def load_image_paths(hr_dir, lr_dir):
    hr_image_paths = sorted([os.path.join(hr_dir, fname) for fname in os.listdir(hr_dir) if fname.lower().endswith('.png')])
    lr_image_paths = sorted([os.path.join(lr_dir, fname) for fname in os.listdir(lr_dir) if fname.lower().endswith('.png')])

    if len(hr_image_paths) != len(lr_image_paths):
        logger.error("Mismatch in number of HR and LR images.")
        raise ValueError("Number of HR and LR images must be the same.")
    
    return list(zip(hr_image_paths, lr_image_paths))

def load_and_preprocess_images(image_pairs, target_hr_size=(500, 500), target_lr_size=(250, 250)):
    hr_images, lr_images = [], []

    for hr_path, lr_path in image_pairs:
        try:
            hr_img = io.imread(hr_path).astype(np.float32) / 255.0
            if hr_img.ndim == 3:
                hr_img = rgb2gray(hr_img)
            hr_img = np.expand_dims(hr_img, axis=-1)
            hr_img = tf.image.resize(hr_img, target_hr_size).numpy()

            lr_img = io.imread(lr_path).astype(np.float32) / 255.0
            if lr_img.ndim == 3:
                lr_img = rgb2gray(lr_img)
            lr_img = np.expand_dims(lr_img, axis=-1)
            lr_img = tf.image.resize(lr_img, target_lr_size).numpy()

            hr_images.append(hr_img)
            lr_images.append(lr_img)
        except Exception as e:
            logger.error(f"Error loading images: {hr_path}, {lr_path} - {e}")
    
    hr_images, lr_images = np.array(hr_images), np.array(lr_images)
    logger.info(f"Loaded {hr_images.shape[0]} image pairs.")
    logger.info(f"LR images shape: {lr_images.shape}")
    logger.info(f"HR images shape: {hr_images.shape}")
    return lr_images, hr_images

# === Evaluation and Visualization Functions ===

def calculate_psnr_ssim(y_true, y_pred):
    psnr_vals = [tf.image.psnr(y_true[i], y_pred[i], max_val=1.0).numpy() for i in range(len(y_true))]
    ssim_vals = [tf.image.ssim(y_true[i], y_pred[i], max_val=1.0).numpy() for i in range(len(y_true))]
    return np.mean(psnr_vals), np.mean(ssim_vals)

def visualize_first_n_predictions(y_lr, y_pred, y_hr, n=5):
    indices = np.random.choice(len(y_lr), n, replace=False)
    for i, idx in enumerate(indices):
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(y_lr[idx, :, :, 0], cmap='gray')
        plt.title('Low-Resolution Input')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(y_pred[idx, :, :, 0], cmap='gray')
        plt.title('Super-Resolved Output')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(y_hr[idx, :, :, 0], cmap='gray')
        plt.title('High-Resolution Ground Truth')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

# === Main Evaluation Workflow ===

def main():
    # Build the model architecture
    logger.info("Building the model architecture...")
    input_shape = (250, 250, 1)  # LR image size
    model = dual_branch_unet(input_shape)

    # Compile the model (compilation is necessary before loading weights)
    model.compile(optimizer='adam', loss=combined_loss, metrics=[psnr_metric, ssim_metric])

    # Load the model weights
    logger.info(f"Loading model weights from '{model_weights_path}'...")
    try:
        model.load_weights(model_weights_path)
        logger.info("Model weights loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model weights: {e}")
        raise e

    # Load test image paths
    logger.info("Loading test image paths...")
    test_image_pairs = load_image_paths(hr_test_dir, lr_test_dir)

    # Load and preprocess test images
    logger.info("Loading and preprocessing test images...")
    lr_test, hr_test = load_and_preprocess_images(test_image_pairs, target_hr_size=target_hr_size, target_lr_size=target_lr_size)

    # Evaluate the model
    logger.info(f"Evaluating the model on {len(lr_test)} test images...")
    sr_test = model.predict(lr_test, batch_size=batch_size, verbose=1)

    avg_psnr, avg_ssim = calculate_psnr_ssim(hr_test, sr_test)
    logger.info(f"Average PSNR on test set: {avg_psnr:.2f} dB")
    logger.info(f"Average SSIM on test set: {avg_ssim:.4f}")

    # Visualize and save the first n predictions
    logger.info(f"Visualizing the first {num_samples_to_visualize} sample predictions...")
    visualize_first_n_predictions(lr_test, sr_test, hr_test, n=num_samples_to_visualize)

if __name__ == "__main__":
    main()
