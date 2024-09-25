# Import necessary libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, UpSampling2D, MaxPooling2D, Add, Multiply, Activation,
    GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Reshape,
    Lambda, Concatenate, BatchNormalization
)
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
from skimage import io
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import logging


# Directories containing the images
hr_train_dir = '/mimer/NOBACKUP/groups/geodl/DeepRockSR-2D/shuffled2D/shuffled2D_train_HR'
lr_train_dir = '/mimer/NOBACKUP/groups/geodl/DeepRockSR-2D/shuffled2D/shuffled2D_train_LR_default_X2'

# Training parameters
batch_size = 4     # Adjusted batch size
epochs = 100
learning_rate = 1e-3

# Set the number of images for train, validation, and test sets
# None to use all  images
num_train = 2000    
num_val = 200       
num_test = 200      

print(f"Batch size: {batch_size}")
print(f"Number of epochs: {epochs}")
print(f"Learning rate: {learning_rate}")
print(f"Number of training images: {num_train}")
print(f"Number of validation images: {num_val}")
print(f"Number of test images: {num_test}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Attention Modules

def channel_attention(input_feature, ratio=8):

    channel = K.int_shape(input_feature)[-1]

    shared_dense_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True)
    shared_dense_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True)

    # Average Pooling
    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_dense_one(avg_pool)
    avg_pool = shared_dense_two(avg_pool)

    # Max Pooling
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_dense_one(max_pool)
    max_pool = shared_dense_two(max_pool)

    # Combine and apply activation
    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    # Apply attention
    output_feature = Multiply()([input_feature, cbam_feature])
    return output_feature

def spatial_attention(input_feature):
 
    # Average and Max Pooling along channel axis using Lambda layers
    avg_pool = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(input_feature)
    max_pool = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(input_feature)

    # Concatenate along channel axis
    concat = Concatenate(axis=-1)([avg_pool, max_pool])

    # Convolution and activation
    cbam_feature = Conv2D(1, kernel_size=7, strides=1, padding='same',
                          activation='sigmoid', kernel_initializer='he_normal',
                          use_bias=False)(concat)

    # Apply attention
    output_feature = Multiply()([input_feature, cbam_feature])
    return output_feature

def cbam_block(input_feature, ratio=8):
    """
    CBAM Block combines channel and spatial attention.
    """
    x = channel_attention(input_feature, ratio)
    x = spatial_attention(x)
    return x


def texture_branch(input_layer):
    """
    Texture branch of the dual-branch U-Net with fewer layers.
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
    Structure branch of the dual-branch U-Net with more layers.
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

    model = Model(inputs=inputs, outputs=fused)
    return model

#  Data Loading and Preprocessing ===

def load_image_paths(hr_dir, lr_dir):
    """
    Loads and pairs high-resolution and low-resolution image paths.
    """
    hr_image_paths = sorted([
        os.path.join(hr_dir, fname)
        for fname in os.listdir(hr_dir)
        if fname.lower().endswith(('.png'))
    ])
    lr_image_paths = sorted([
        os.path.join(lr_dir, fname)
        for fname in os.listdir(lr_dir)
        if fname.lower().endswith(('.png'))
    ])

    # Ensure that the lists are of the same length
    if len(hr_image_paths) != len(lr_image_paths):
        logger.error("Mismatch in number of HR and LR images.")
        raise ValueError("Number of HR and LR images must be the same.")

    # Pair HR and LR image paths
    image_pairs = list(zip(hr_image_paths, lr_image_paths))

    # Shuffle the image pairs
    shuffle(image_pairs, random_state=42)

    return image_pairs

def split_image_pairs(image_pairs, num_train, num_val, num_test):
    """
    Splits image pairs into train, validation, and test sets based on specified numbers.
    """
    total_images = len(image_pairs)
    specified_sum = 0
    if num_train is not None:
        specified_sum += num_train
    if num_val is not None:
        specified_sum += num_val
    if num_test is not None:
        specified_sum += num_test

    if specified_sum > total_images:
        raise ValueError("The sum of train, val, and test images exceeds the total number of images.")

    train_pairs = []
    val_pairs = []
    test_pairs = []
    remaining_pairs = image_pairs.copy()

    # Assign training images
    if num_train is not None:
        train_pairs = remaining_pairs[:num_train]
        remaining_pairs = remaining_pairs[num_train:]

    # Assign validation images
    if num_val is not None:
        val_pairs = remaining_pairs[:num_val]
        remaining_pairs = remaining_pairs[num_val:]

    # Assign test images
    if num_test is not None:
        test_pairs = remaining_pairs[:num_test]
        remaining_pairs = remaining_pairs[num_test:]

    # Assign remaining images to the first set that has None
    if remaining_pairs:
        if num_train is None:
            train_pairs.extend(remaining_pairs)
        elif num_val is None:
            val_pairs.extend(remaining_pairs)
        elif num_test is None:
            test_pairs.extend(remaining_pairs)
        else:
            logger.warning("Some images were not assigned to any set. Consider adjusting your numbers.")

    return train_pairs, val_pairs, test_pairs

def load_and_preprocess_images(image_pairs, target_hr_size=(500, 500), target_lr_size=(250, 250)):
    """
    Loads and preprocesses images from given image pairs.
    """
    hr_images = []
    lr_images = []

    for hr_path, lr_path in image_pairs:
        try:
            # Load HR image
            hr_img = io.imread(hr_path).astype(np.float32) / 255.0
            if hr_img.ndim == 3:
                hr_img = rgb2gray(hr_img)
            hr_img = np.expand_dims(hr_img, axis=-1)
            hr_img = tf.image.resize(hr_img, target_hr_size).numpy()

            # Load LR image
            lr_img = io.imread(lr_path).astype(np.float32) / 255.0
            if lr_img.ndim == 3:
                lr_img = rgb2gray(lr_img)
            lr_img = np.expand_dims(lr_img, axis=-1)
            lr_img = tf.image.resize(lr_img, target_lr_size).numpy()

            hr_images.append(hr_img)
            lr_images.append(lr_img)
        except Exception as e:
            logger.error(f"Error loading images: {hr_path}, {lr_path} - {e}")
            continue  # Skip problematic images

    hr_images = np.array(hr_images)
    lr_images = np.array(lr_images)

    print(f"Loaded {hr_images.shape[0]} image pairs.")
    print(f"LR images shape: {lr_images.shape}")
    print(f"HR images shape: {hr_images.shape}")

    return lr_images, hr_images

# Load and Preprocess the Images 

# Load image paths
all_image_pairs = load_image_paths(hr_train_dir, lr_train_dir)

# Split image pairs into train, val, and test sets
train_pairs, val_pairs, test_pairs = split_image_pairs(all_image_pairs, num_train, num_val, num_test)

print(f"Number of training image pairs: {len(train_pairs)}")
print(f"Number of validation image pairs: {len(val_pairs)}")
print(f"Number of test image pairs: {len(test_pairs)}")

# Load and preprocess images for each set
lr_train, hr_train = load_and_preprocess_images(train_pairs)
lr_val, hr_val = load_and_preprocess_images(val_pairs)
lr_test, hr_test = load_and_preprocess_images(test_pairs)

# === Model Instantiation ===

input_shape = (250, 250, 1)  # LR image size
model = dual_branch_unet(input_shape)

def combined_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    return mse + mae


# PSNR and SSIM Metrics 

def psnr_metric(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def ssim_metric(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=1.0)

#  Compile Model 

optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss=combined_loss, metrics=[psnr_metric, ssim_metric])
model.summary()

#  Callbacks 

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'models/best_model_1.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    verbose=1,
    min_lr=1e-7
)

# Combine callbacks
callbacks = [early_stopping, checkpoint, reduce_lr]

# Training

history = model.fit(
    lr_train, hr_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(lr_val, hr_val),
    callbacks=callbacks,
    verbose=1
)

#  Plotting Training History 

def plot_training_history(history, save_path='plots/training_history_1.png'):
    """
    Plots the training and validation loss over epochs and saves the plot to a file.

    Args:
        history: Keras History object.
        save_path (str): File path to save the plot image.
    """
    plt.figure(figsize=(12, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Combined Loss (MSE + MAE)')
    plt.legend()
    plt.grid(True)

    # Plot PSNR
    plt.subplot(1, 2, 2)
    plt.plot(history.history['psnr_metric'], label='Training PSNR')
    plt.plot(history.history['val_psnr_metric'], label='Validation PSNR')
    plt.title('PSNR Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training history plot saved to '{save_path}'")


plot_training_history(history)

# === Evaluation and Visualization ===
def visualize_predictions(model, lr_images, hr_images, num_samples=5, save_dir='plots'):
    """
    Visualizes the model's predictions alongside the LR inputs and HR ground truths,
    and saves the plots to files.

    Args:
        model: Trained Keras model.
        lr_images (numpy.ndarray): Array of LR input images.
        hr_images (numpy.ndarray): Array of HR ground truth images.
        num_samples (int): Number of samples to visualize.
        save_dir (str): Directory where to save the plots.
    """
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    indices = np.random.choice(len(lr_images), num_samples, replace=False)

    for i, idx in enumerate(indices):
        lr = lr_images[idx]
        hr = hr_images[idx]
        sr = model.predict(np.expand_dims(lr, axis=0))[0]  # Super-resolved image

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(lr[:, :, 0], cmap='gray')
        plt.title('Low-Resolution Input', fontsize=14)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(sr[:, :, 0], cmap='gray')
        plt.title('Super-Resolved Output', fontsize=14)
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(hr[:, :, 0], cmap='gray')
        plt.title('High-Resolution Ground Truth', fontsize=14)
        plt.axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # Adjust top margin to make room for titles
        save_path = os.path.join(save_dir, f'prediction_{i+1}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved prediction visualization to '{save_path}'")

# Evaluation on Test Set

def evaluate_model(model, lr_images, hr_images):
    """
    Evaluates the model using PSNR and SSIM metrics.
    """
    sr_images = model.predict(lr_images)

    psnr_values = []
    ssim_values = []

    for sr, hr in zip(sr_images, hr_images):
        psnr_val = tf.image.psnr(hr, sr, max_val=1.0).numpy()
        ssim_val = tf.image.ssim(hr, sr, max_val=1.0).numpy()
        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)

    print(f"Average PSNR on Test Set: {np.mean(psnr_values):.2f} dB")
    print(f"Average SSIM on Test Set: {np.mean(ssim_values):.4f}")

# Evaluate the model on the test set
evaluate_model(model, lr_test, hr_test)

# Visualize some predictions on the test set and save the plots
visualize_predictions(model, lr_test, hr_test, num_samples=5, save_dir='plots')

# Saving the Final Model
model.save('models/dual_branch_unet_with_attention_final_1.keras')
print("Model saved to 'dual_branch_unet_with_attention_final.keras'")
