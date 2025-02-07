import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

# Function to calculate Bit Error Rate (BER)
def bit_error_rate(original_watermark, extracted_watermark):
    return np.sum(original_watermark != extracted_watermark) / np.size(original_watermark)

# Function to calculate Correlation Coefficient
def correlation_coefficient(original, extracted):
    return np.corrcoef(original.flatten(), extracted.flatten())[0, 1]

# Function to calculate Normalized Cross-Correlation (NCC)
def normalized_cross_correlation(original, extracted):
    original = original.astype(np.float32)
    extracted = extracted.astype(np.float32)
    return np.sum(original * extracted) / (np.sqrt(np.sum(original**2)) * np.sqrt(np.sum(extracted**2)))

# Load the original and watermarked images (assuming grayscale images)
host_image_path = 'watermarked_B1.bmp'
host_image = cv2.imread(host_image_path, cv2.IMREAD_GRAYSCALE)

# List of attack images paths
attack_images_paths = ['watermarked_B1_ROT_45.bmp', 'watermarked_B1_RNDDIST_0.95.bmp', 'watermarked_B1_PSNR_90.bmp','watermarked_B1_NOISE_40.bmp','watermarked_B1_NOISE_20.bmp','watermarked_B1_MEDIAN_9.bmp','watermarked_B1_JPEG_90.bmp','watermarked_B1_CROP_75.bmp','watermarked_B1_CONV_1.bmp','watermarked_B1_AFFINE_3.bmp','blurred.bmp','bending_attack.bmp','attacked_salt_pepper.bmp','attacked_gaussian_blur.bmp'] # Add more paths as needed

# Iterate over each attack image
for i, attack_image_path in enumerate(attack_images_paths):
    attack_image = cv2.imread(attack_image_path, cv2.IMREAD_GRAYSCALE)

    # Calculate metrics
    psnr_value = peak_signal_noise_ratio(host_image, attack_image)
    ssim_value, _ = structural_similarity(host_image, attack_image, full=True)
    mse_value = mean_squared_error(host_image, attack_image)
    ncc_value = normalized_cross_correlation(host_image, attack_image)

    # Print results for each attack
    if i == 0:
        print(f'--- ROT_45 ---')
    elif i == 1:
        print(f'--- RNDDIST_0.95 ---')
    elif i == 2:
        print(f'--- PSNR_90 ---')
    elif i == 3:
        print(f'--- NOISE_40 ---')
    elif i == 4:
        print(f'--- NOISE_20 ---')
    elif i == 5:
        print(f'--- MEDIAN_9 ---')
    elif i == 6:
        print(f'--- JPEG_90 ---')
    elif i == 7:
        print(f'--- CROP_75 ---')
    elif i == 8:
        print(f'--- CONV_1 ---')
    elif i == 9:
        print(f'--- AFFINE_3 ---')
    elif i == 10:
        print(f'--- BLURRED ---')
    elif i == 11:
        print(f'--- BENDING_ATTACK ---')
    elif i == 12:
        print(f'--- SALT_PEPPER ---')
    elif i == 13:
        print(f'--- GAUSSIAN_BLUR ---')
    else:
        print(f'--- Attack {i+1} ---')
    print(f'Peak Signal-to-Noise Ratio (PSNR): {psnr_value:.2f} dB')
    print(f'Structural Similarity Index (SSIM): {ssim_value:.4f}')
    print(f'Mean Squared Error (MSE): {mse_value:.6f}')
    print(f'Normalized Cross-Correlation (NCC): {ncc_value:.4f}')
    print('---------------------\n')