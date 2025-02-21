import os
import cv2
import matplotlib.pyplot as plt

def load_and_display_first_frame(video_path):
    """Load and display the first frame of a given video path."""
    first_frame_path = os.path.join(video_path, "000001.jpg")
    image = cv2.imread(first_frame_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.imshow(image)
    plt.axis("off")
    plt.title("First Frame of "+video_path[-17:])
    plt.show()

def visualize_frame_distribution(video_dirs):
    """Count number of frames in each sequence and visualize the distribution."""
    frame_counts = {path[-17:]: len(os.listdir(path)) for path in video_dirs}
    print(frame_counts)
    
    plt.bar(frame_counts.keys(), frame_counts.values(), color='blue')
    plt.xlabel("Video Sequence")
    plt.ylabel("Number of Frames")
    plt.title("Frame Distribution Across Videos")
    plt.show()

# Define paths
video_dirs = [
    "train/MOT17-11-SDP/img1",
    "train/MOT17-13-SDP/img1"
]

# Execute functions
load_and_display_first_frame(video_dirs[0])
load_and_display_first_frame(video_dirs[1])
visualize_frame_distribution(video_dirs)