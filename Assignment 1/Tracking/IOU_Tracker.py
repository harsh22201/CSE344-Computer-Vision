import numpy as np

class IOUTracker:
    def __init__(self, iou_threshold=0.7):
        """
        Initialize the IOU Tracker.

        Args:
            iou_threshold (float): Minimum IoU score to match a detection to a tracker.
        """
        self.iou_threshold = iou_threshold
        self.trackers = [] # Store active trackers as [xmin, ymin, xmax, ymax, track_id, class, score]
        self.next_id = 0    # Unique ID for the next tracker

    def _compute_iou(self, box1, box2):
        """
        Compute Intersection-over-Union (IoU) between two bounding boxes.
        Each box is in format [x_min, y_min, x_max, y_max].
        """

        # Compute intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        # Compute union area
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def update(self, detections):
        """
        Update the trackers with new detections.
        - Input Format : Numpy Array of detections in the format: [xmin,ymin,xmax,ymax,score,class]
        - Return the updated list of trackers in the format: [xmin, ymin, xmax, ymax, track id, class, score].
        """
        updated_trackers = []
        # Perform greedy matching using IOU: compare each tracker with new detections.
        for detection in detections:
            xmin, ymin, xmax, ymax, score, cls = detection
            matched = False
            for tracker in self.trackers:
                # If IOU > threshold, match the detection to the tracker.
                if self._compute_iou(tracker[:4], [xmin, ymin, xmax, ymax]) > self.iou_threshold:
                    tracker[:4] = [xmin, ymin, xmax, ymax] # Update the bounding box
                    tracker[6] = score # Update the score
                    updated_trackers.append(tracker) # Keep the matched tracker
                    matched = True
                    break
            # If no match is found for a detection, create a new tracker.
            if not matched:
                updated_trackers.append([xmin, ymin, xmax, ymax, self.next_id, cls, score])
                self.next_id += 1

        # Remove trackers that don't match any detection from the previous frame by updating self.trackers
        self.trackers = updated_trackers
        return self.trackers