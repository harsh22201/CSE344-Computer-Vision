import cv2
import numpy as np
import pandas as pd
import pickle
import torch

import motmetrics as mm

import os

import time
from byte.byte_tracker import BYTETracker
from IOU_Tracker import IOUTracker

def get_image_frames(image_dir):
    """
    Reads frames (images) from a directory.

    Args:
        image_dir (str): Path to the directory containing images.

    Returns:
        list: A list of image frames (numpy arrays).
    """
    
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    if not image_files:
        print(f"No images found in {image_dir}.")
        return []
    
    frames = []
    for file in image_files:
        frame = cv2.imread(file)
        if frame is None:
            print(f"Error loading image: {file}")
        else:
            frames.append(frame)
    
    return frames


def load_mot_detections(det_path):
    """
    Load MOT format detections from a file.

    Args:
        det_path (str): Path to the detection file.
        det.txt format = [frame,id,x_left,y_bottom,w,h,score]

    Returns:
        list: A list of detections, where each detection follows the format:
            [frame, xmin, ymin, xmax, ymax, score, class]
    """
    detections = []

    with open(det_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')

            # Extract values from `parts` and convert them to appropriate data types
            frame = int(parts[0])  # Extract frame number
            x = float(parts[2])    # Extract x coordinate
            y = float(parts[3])    # Extract y coordinate
            w = float(parts[4])    # Extract width
            h = float(parts[5])    # Extract height
            score = float(parts[6])  # Extract confidence score

            # Skip invalid detections (e.g., if x < 0 or y < 0)
            if x < 0 or y < 0:
                continue

            # Calculate xmin, ymin, xmax, ymax
            xmin = x
            ymin = y
            xmax = x + w
            ymax = y + h

            cls = 0  # Default class (can be modified if needed)

            # Append the detection in the correct format
            detections.append([frame, xmin, ymin, xmax, ymax, score, cls])

    return detections


def real_time_dataset(frames, detections, fps=30):
    time_per_frame = 1 / fps
    for frame_idx, frame in enumerate(frames):
        # Get detections for the current frame
        frame_detections = [d for d in detections if d[0] == frame_idx + 1]
        yield frame, frame_detections  # Yield current frame and its detections
        time.sleep(time_per_frame) 


#Usable with IOU tracker or ByteTracker
def run_tracker(tracker, frames, detections, fps=30):
    """
    Run the tracker on the given frames and detections, simulating real-time input.

    Args:
        tracker: Tracker object (IOUTracker or ByteTracker).
        frames (list): List of frames (images) as numpy arrays.
        detections (list): List of detections in MOT format.
        fps (int): Frames per second for simulating real-time processing.

    Returns:
        list: List of tracked objects in MOT format.
    """
    tracked_objects = []

    # Initialize video writer for output
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    out = cv2.VideoWriter("tracking.mp4", fourcc, fps, (width, height))

    # Simulate real-time frame input using the generator
    frame_gen = real_time_dataset(frames, detections, fps)

    for frame_idx, (frame, frame_detections) in enumerate(frame_gen):
        if len(frame_detections) == 0:
            continue

        # Convert detections to numpy array
        detection_array = np.array([d[1:] for d in frame_detections])

        # Update tracker
        online_tracks = tracker.update(detection_array)

        # Extract tracked object information
        for track in online_tracks:
            xmin, ymin, xmax, ymax, track_id, cls, score = track
            # track_id = None
            
            x = xmin
            y = ymin
            w = xmax - xmin
            h = ymax - ymin
            # cls = None
            # score = None

            # Append tracked object to the result list
            tracked_objects.append([frame_idx + 1, track_id, x, y, w, h, score, cls])

            # Draw bounding box and track ID on the frame
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Write frame to output video
        out.write(frame)
    
    # Release video writer
    out.release()
    return tracked_objects

def evaluate_tracking(gt_path, tracked_objects):
    gt_data = pd.read_csv(gt_path, header=None, names=["frame", "id", "x", "y", "w", "h", "conf", "class", "visibility"])
    print(len(gt_data))
    gt_data = gt_data[gt_data["conf"] != 0]
    print(len(gt_data))
    gt_data = gt_data[gt_data["y"] != -1]
    track_df = pd.DataFrame(tracked_objects, columns=["frame", "id", "x", "y", "w", "h","conf","class"])

    track_df.to_csv('output.txt',sep=',',index=False)

    acc = mm.MOTAccumulator(auto_id=True)
    for frame in sorted(gt_data["frame"].unique()):
        gt_frame = gt_data[gt_data["frame"] == frame]
        pred_frame = track_df[track_df["frame"] == frame]

        gt_ids = gt_frame["id"].values
        pred_ids = pred_frame["id"].values
        gt_boxes = gt_frame[["x", "y", "w", "h"]].values
        pred_boxes = pred_frame[["x", "y", "w", "h"]].values

        distances = mm.distances.iou_matrix(gt_boxes,pred_boxes,max_iou=0.5)

        acc.update(gt_ids, pred_ids, distances)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name="Overall")
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    mota_score = summary.loc["Overall", "mota"]
    
    return mota_score


if __name__ == "__main__":
    # Paths to the video and ground truth
    
    image_path = "train/MOT17-13-SDP/img1"
    gt_path = "train/MOT17-13-SDP/gt/gt.txt"
    det_path= "train/MOT17-13-SDP/det/det.txt"
    
    frames = get_image_frames(image_path)
    print(f"Loaded {len(frames)} frames from video.")

    detections = load_mot_detections(det_path)
    print(f"Detections generated: {len(detections)}")
    
    trackers = {
        "Byte": BYTETracker(),  # Default parameters
        "IouTracker": IOUTracker(iou_threshold=0.8)  # IOU threshold set to 0.8
    }
    
    for name, tracker in trackers.items():
        # Get the tracked objects by using run_tracker.
        tracked_objects = run_tracker(tracker, frames, detections)
        print(f"{name} Tracking results generated: {len(tracked_objects)}")
        
        # Save the tracked objects to a pickle file
        with open(f"{name}.pkl", "wb") as f:
            pickle.dump(tracked_objects, f)
            print(f"Saved {name} tracking results to {name}.pkl")
        
        # Evaluate tracking
        mota_score = evaluate_tracking(gt_path, tracked_objects)
        print(f"{name} MOTA Score: {mota_score:.4f}")