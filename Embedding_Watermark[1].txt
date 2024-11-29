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
from moviepy.editor import VideoFileClip

def embed_watermark(video_path, watermark_path, output_path, alpha):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video file '{video_path}'.")
        return False

    watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
    if watermark is None:
        print(f"Error: Could not load watermark image '{watermark_path}'.")
        return False

    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_rate = int(video.get(cv2.CAP_PROP_FPS))


    # Perform watermarking on each channel
    wavelet_filter = 'haar'
    decomposition_level = 3
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height))
    if not writer.isOpened():
        print(f"Error: Could not create video writer for '{output_path}'.")
        return False

    while True:
        ret, frame = video.read()
        if not ret:
            break
        # Split into color channels
        b, g, r = cv2.split(frame)

# Apply RDWT decomposition to each channel
        coeffs_b = pywt.wavedec2(b, wavelet_filter, level=decomposition_level)
        coeffs_g = pywt.wavedec2(g, wavelet_filter, level=decomposition_level)
        coeffs_r = pywt.wavedec2(r, wavelet_filter, level=decomposition_level)

# Embed watermark in the LL subband of each channel
        LL_b = coeffs_b[0]
        LL_g = coeffs_g[0]
        LL_r = coeffs_r[0]

# Resize watermark to match subband size
        watermark_resized = cv2.resize(watermark, (LL_b.shape[1], LL_b.shape[0]), interpolation=cv2.INTER_CUBIC)

# Normalize watermark values
        watermark_normalized = (watermark_resized / 255.0) * np.max([np.max(LL_b), np.max(LL_g), np.max(LL_r)])

# Embed watermark
        LL_b += alpha * watermark_normalized
        LL_g += alpha * watermark_normalized
        LL_r += alpha * watermark_normalized

# Update coeffs lists
        coeffs_b[0] = LL_b
        coeffs_g[0] = LL_g
        coeffs_r[0] = LL_r

# Apply inverse RDWT to obtain watermarked channels
        watermarked_b = pywt.waverec2(coeffs_b, wavelet_filter)
        watermarked_g = pywt.waverec2(coeffs_g, wavelet_filter)
        watermarked_r = pywt.waverec2(coeffs_r, wavelet_filter)

# Ensure values are within valid range
        watermarked_b = np.clip(watermarked_b, 0, 255).astype(np.uint8)
        watermarked_g = np.clip(watermarked_g, 0, 255).astype(np.uint8)
        watermarked_r = np.clip(watermarked_r, 0, 255).astype(np.uint8)

# Merge channels back
        watermarked_frame = cv2.merge((watermarked_b, watermarked_g, watermarked_r))
        writer.write(watermarked_frame)
    # Release resources
    video.release()
    writer.release()

    print("Watermark embedding successful!")

embed_watermark("/content/drive/MyDrive/Colab Notebooks/sample.mp4", "/content/secret11.png", "watermarked.mp4", 0.1)

from moviepy.editor import VideoFileClip
clip = VideoFileClip("watermarked.mp4")
clip.write_videofile("compressed_video.mp4", codec="libx264", bitrate="5000k")  # Adjust bitrate as needed

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2

# Load original frame from video
original_video = cv2.VideoCapture("/content/drive/MyDrive/Colab Notebooks/sample.mp4")  # Replace with original video path
ret, original_frame = original_video.read()
original_video.release()

# Load watermarked frame from video
watermarked_video = cv2.VideoCapture("watermarked.mp4")  # Replace with watermarked video path
ret, watermarked_frame = watermarked_video.read()
watermarked_video.release()

# Convert frames to grayscale for SSIM/PSNR calculation
original_frame_gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
watermarked_frame_gray = cv2.cvtColor(watermarked_frame, cv2.COLOR_BGR2GRAY)


# Calculate SSIM
ssim_value = ssim(original_frame_gray, watermarked_frame_gray, data_range=watermarked_frame.max() - watermarked_frame.min())

# Calculate PSNR
psnr_value = psnr(original_frame_gray, watermarked_frame_gray, data_range=watermarked_frame_gray.max() - watermarked_frame_gray.min())

print(f"SSIM: {ssim_value}")
print(f"PSNR: {psnr_value}")

# Load watermarked frame from video
watermarked_video1 = cv2.VideoCapture("compressed_video.mp4")  # Replace with watermarked video path
ret, watermarked_frame1 = watermarked_video1.read()
watermarked_video1.release()

# Convert frames to grayscale for SSIM/PSNR calculation

watermarked_frame_gray1 = cv2.cvtColor(watermarked_frame1, cv2.COLOR_BGR2GRAY)


# Calculate SSIM
ssim_value1 = ssim(original_frame_gray, watermarked_frame_gray1, data_range=watermarked_frame.max() - watermarked_frame.min())

# Calculate PSNR
psnr_value1 = psnr(original_frame_gray, watermarked_frame_gray1, data_range=watermarked_frame_gray.max() - watermarked_frame_gray.min())

print(f"SSIM: {ssim_value1}")
print(f"PSNR: {psnr_value1}")
