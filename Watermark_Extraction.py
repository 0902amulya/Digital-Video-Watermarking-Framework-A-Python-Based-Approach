"""
import required modules using :
pip install PyWavelets
pip install numpy
pip install opencv-python
pip install moviepy
"""
import cv2
import pywt
import numpy as np
def extract_watermark(watermarked_video_path, output_watermark_path, alpha, wavelet_filter='haar', decomposition_level=3):
    video = cv2.VideoCapture(watermarked_video_path)
    if not video.isOpened():
        print(f"Error: Could not open video file '{watermarked_video_path}'.")
        return False

    ret, frame = video.read()  # Read the first frame
    if not ret:
        print("Error: Could not read a frame from the video.")
        return False

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)

    # Apply RDWT decomposition
    coeffs_watermarked = pywt.wavedec2(frame_gray, wavelet_filter, level=decomposition_level)
    LL_watermarked = coeffs_watermarked[0]

    # Assuming you have the original frame for reference
    original_video = cv2.VideoCapture("/content/drive/MyDrive/Colab Notebooks/sample.mp4")
    ret, original_frame = original_video.read()
    original_video.release()

    original_frame_gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
    original_frame_gray = cv2.GaussianBlur(original_frame_gray, (5, 5), 0)

    coeffs_original = pywt.wavedec2(original_frame_gray, wavelet_filter, level=decomposition_level)
    LL_original = coeffs_original[0]

    # Extract watermark from LL subband
    extracted_watermark = (LL_watermarked - LL_original) / alpha

    # Normalize the extracted watermark
    extracted_watermark = cv2.normalize(extracted_watermark, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Save the extracted watermark
    cv2.imwrite(output_watermark_path, extracted_watermark)

    video.release()
    print("Watermark extraction successful!")
    return True


extract_watermark("/content/watermarked (32).mp4", "extracted_watermark.png", 0.1)
extracted_watermark = cv2.imread("extracted_watermark.png", cv2.IMREAD_GRAYSCALE)

#extracted_watermark1=cv2.imread("extracted_watermark_enhanced.png", cv2.IMREAD_GRAYSCALE)
# Load the original watermark (assuming it's available)
#original_watermark = cv2.imread("/content/drive/MyDrive/Colab Notebooks/secret2.png", cv2.IMREAD_GRAYSCALE)



def calculate_nc(original_watermark, extracted_watermark):
  """Calculates the Normalized Correlation (NC) between two watermarks.

  Args:
    original_watermark: The original watermark as a NumPy array (normalized to 0-1).
    extracted_watermark: The extracted watermark as a NumPy array (normalized to 0-1).

  Returns:
    The NC value, a float between -1 and 1.
  """
    # Normalize watermarks to the range [0, 1]
  original_normalized = (original_watermark - np.min(original_watermark)) / (np.max(original_watermark) - np.min(original_watermark))
  extracted_normalized = (extracted_watermark - np.min(extracted_watermark)) / (np.max(extracted_watermark) - np.min(extracted_watermark))

  # Ensure both watermarks are flattened arrays
  original_flat = original_normalized.flatten()
  extracted_flat = extracted_normalized.flatten()
  smaller_size = min(len(original_flat), len(extracted_flat))
  original_flat = original_flat[:smaller_size]
  extracted_flat = extracted_flat[:smaller_size]
  # Calculate NC
  numerator = np.sum(original_flat * extracted_flat)
  denominator = np.sqrt(np.sum(original_flat**2) * np.sum(extracted_flat**2))


  if denominator == 0:
    return 0  # Handle potential division by zero

  return numerator / denominator
   # Call calculate_nc with normalized arrays



original_watermark = cv2.imread("/content/secret11.png", cv2.IMREAD_GRAYSCALE)
extracted_watermark_enhanced = cv2.equalizeHist(extracted_watermark)
# Apply contrast enhancement (e.g., histogram equalization)
cv2.imwrite("extracted_watermark_enhanced.png", extracted_watermark_enhanced)
print("Extracted watermark is enhanced")
nc_value = calculate_nc(original_watermark, extracted_watermark_enhanced)
print("Normalized Correlation (NC):", nc_value)
extracted_watermark_denoised = cv2.GaussianBlur(extracted_watermark_enhanced, (5, 5), 0)
# (5, 5) is the kernel size, 0 is the standard deviation in both directions
print("Extracted watermark is denoised")
nc_value = calculate_nc(original_watermark, extracted_watermark_denoised)
cv2.imwrite("extracted_watermark_denoised.png", extracted_watermark_denoised)
print("Normalized Correlation (NC):", nc_value)
# Calculate NC
# Load the original watermark (assuming it's available)

nc_value = calculate_nc(original_watermark, extracted_watermark)
print("Normalized Correlation (NC):", nc_value)





