import subprocess
import ffmpeg
import cv2
import matplotlib.pyplot as plt
import skvideo.io
import numpy as np

import skvideo.motion
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm

np.float = np.float64
np.int = np.int_


def convert_yuv_to_avi(yuv_file_path, width, height, output_file_path, chroma_format='yuv420p'):
    """
    Convert a CIF YUV file to a non-compressed raw video file in AVI format.

    :param yuv_file_path: Path to the input YUV file.
    :param width: Width of the video frame.
    :param height: Height of the video frame.
    :param output_file_path: Path to the output AVI file.
    :param chroma_format: Chroma subsampling format (default is 'yuv420p').
    """
    # Build the ffmpeg command
    command = [
        'ffmpeg',
        '-s', f'{width}x{height}',  # Set the frame size
        '-pix_fmt', chroma_format,  # Set the pixel format
        '-i', yuv_file_path,  # Input file
        '-c:v', 'rawvideo',  # Output codec (raw video)
        '-an',  # No audio
        output_file_path  # Output file
    ]

    # Run the ffmpeg command
    subprocess.run(command, check=True)
    print(f"Converted {yuv_file_path} to {output_file_path}")


def extract_frames(video_path, idx):
    """
    Extracts specified frames from the video.

    Parameters:
    - video_path: Path to the video file.
    - frame_indices: List of frame indices to extract.

    Returns:
    - List of extracted frames.
    """
    videogen = skvideo.io.vreader(video_path)

    # Choose the frame number to extract
    frame_number = idx

    # Read frames one by one
    for i, frame in enumerate(videogen):
        if i == frame_number:
            selected_frame = frame
            break

    # Convert the frame to RGB if necessary (skvideo gives frames in RGB format)
    if selected_frame.shape[2] == 3:
        frame_rgb = selected_frame
    else:
        frame_rgb = np.repeat(selected_frame[:, :, np.newaxis], 3, axis=2)

    return frame_rgb

if __name__ == '__main__':
    width = 1920  # CIF width
    height = 1080  # CIF height
    #yuv_file_path = r"C:\final_project_eran_nadav\semantic-segmentation-main\data\elbit_dataset\videofile-005.yuv"
    file_path = r"C:\final_project_eran_nadav\semantic-segmentation-main\data\elbit_dataset\videofile-005.avi"
    #output_file_path = convert_yuv_to_avi(yuv_file_path, width, height, output_file_path, chroma_format='yuv420p')
    frame_indices = [21, 42]
    extract_frames(file_path, frame_indices)