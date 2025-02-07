from PIL import Image
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog
import matplotlib.pyplot as plt
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve

def load_and_preprocess(image_path):
    try:
        pil_img = Image.open(image_path)
        img = np.array(pil_img)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (256, 256))
        img = cv2.equalizeHist(img)
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def extract_lbp_features(image, n_points=8, radius=1, method='uniform'):
    lbp = local_binary_pattern(image, n_points, radius, method)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_hog_features(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    fd = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
             cells_per_block=cells_per_block, visualize=False)
    return fd

def lbp_hog_fusion(image):
    lbp_features = extract_lbp_features(image)
    hog_features = extract_hog_features(image)
    combined_features = np.concatenate((lbp_features, hog_features))
    return combined_features

def reduce_features(features, n_components=256):
    variances = np.var(features, axis=0)
    top_indices = np.argsort(variances)[-n_components:]
    reduced_features = features[:, top_indices]
    return reduced_features.flatten()

def calculate_similarity(feature_vec1, feature_vec2):
    return cosine_similarity([feature_vec1], [feature_vec2])[0][0]

def calculate_far_frr(genuine_scores, impostor_scores):
    labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    scores = np.concatenate([genuine_scores, impostor_scores])

    fpr, tpr, thresholds = roc_curve(labels, scores)

    far = fpr  # False Acceptance Rate
    frr = 1 - tpr  # False Rejection Rate

    return far, frr, thresholds

# Main execution
if __name__ == "__main__":
    # قائمة بأسماء الملفات المقدمة
    image_paths = [
        "101_1.tif", "101_2.tif", "101_3.tif", "101_4.tif", "101_5.tif", "101_6.tif", "101_7.tif", "101_8.tif",
        "102_1.tif", "102_2.tif", "102_3.tif", "102_4.tif", "102_5.tif", "102_6.tif", "102_7.tif", "102_8.tif",
        "103_1.tif", "103_2.tif", "103_3.tif", "103_4.tif", "103_5.tif", "103_6.tif", "103_7.tif", "103_8.tif",
        "104_1.tif", "104_2.tif", "104_3.tif", "104_4.tif", "104_5.tif", "104_6.tif", "104_7.tif", "104_8.tif",
        
    ]

    # إنشاء labels بناءً على الرقم الأول لكل صورة
    labels = [int(img.split('_')[0]) for img in image_paths]

    lbp_scores_genuine, lbp_scores_impostor = [], []
    hog_scores_genuine, hog_scores_impostor = [], []
    combined_scores_genuine, combined_scores_impostor = [], []
    reduced_scores_genuine, reduced_scores_impostor = [], []

    for i, image_path in enumerate(image_paths):
        img = load_and_preprocess(image_path)

        lbp_features = extract_lbp_features(img)
        hog_features = extract_hog_features(img)
        combined_features = lbp_hog_fusion(img)
        reduced_features = reduce_features(combined_features.reshape(1, -1))

        for j, compare_path in enumerate(image_paths):
            if i != j:  # Avoid comparing the same image
                compare_img = load_and_preprocess(compare_path)

                compare_lbp = extract_lbp_features(compare_img)
                compare_hog = extract_hog_features(compare_img)
                compare_combined = lbp_hog_fusion(compare_img)
                compare_reduced = reduce_features(compare_combined.reshape(1, -1))

                lbp_score = calculate_similarity(lbp_features, compare_lbp)
                hog_score = calculate_similarity(hog_features, compare_hog)
                combined_score = calculate_similarity(combined_features, compare_combined)
                reduced_score = calculate_similarity(reduced_features, compare_reduced)

                if labels[i] == labels[j]:  # Genuine pairs
                    lbp_scores_genuine.append(lbp_score)
                    hog_scores_genuine.append(hog_score)
                    combined_scores_genuine.append(combined_score)
                    reduced_scores_genuine.append(reduced_score)
                else:  # Impostor pairs
                    lbp_scores_impostor.append(lbp_score)
                    hog_scores_impostor.append(hog_score)
                    combined_scores_impostor.append(combined_score)
                    reduced_scores_impostor.append(reduced_score)

    # Calculate FAR and FRR for LBP, HOG, Combined, and Reduced
    lbp_far, lbp_frr, lbp_thresholds = calculate_far_frr(lbp_scores_genuine, lbp_scores_impostor)
    hog_far, hog_frr, hog_thresholds = calculate_far_frr(hog_scores_genuine, hog_scores_impostor)
    combined_far, combined_frr, combined_thresholds = calculate_far_frr(combined_scores_genuine, combined_scores_impostor)
    reduced_far, reduced_frr, reduced_thresholds = calculate_far_frr(reduced_scores_genuine, reduced_scores_impostor)

    # Plot FAR vs FRR for comparison
    plt.figure(figsize=(10, 8))
    plt.plot(lbp_far, lbp_frr, label='LBP')
    plt.plot(hog_far, hog_frr, label='HOG')
    plt.plot(combined_far, combined_frr, label='Combined')
    plt.plot(reduced_far, reduced_frr, label='Reduced')

    plt.xlabel('FAR (False Acceptance Rate)')
    plt.ylabel('FRR (False Rejection Rate)')
    plt.title('FAR vs FRR for LBP, HOG, Combined, and Reduced Features')
    plt.legend()
    plt.show()

    # Find and print EER for each method (where FAR and FRR are closest)
    lbp_eer = lbp_far[np.nanargmin(np.abs(lbp_far - lbp_frr))]
    hog_eer = hog_far[np.nanargmin(np.abs(hog_far - hog_frr))]
    combined_eer = combined_far[np.nanargmin(np.abs(combined_far - combined_frr))]
    reduced_eer = reduced_far[np.nanargmin(np.abs(reduced_far - reduced_frr))]

    print(f"LBP EER: {lbp_eer:.4f}")
    print(f"HOG EER: {hog_eer:.4f}")
    print(f"Combined EER: {combined_eer:.4f}")
    print(f"Reduced EER: {reduced_eer:.4f}")
