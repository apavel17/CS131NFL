import cv2
import os

video_path = "game.mp4"

start_time_sec = 1696

num_frames = 1

workspace_dir = os.path.abspath(os.path.dirname(__file__)) 
output_dir = os.path.join(workspace_dir, "play02") 
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

cap.set(cv2.CAP_PROP_POS_MSEC, start_time_sec * 1000)

frame_count = 0

while cap.isOpened() and frame_count < num_frames:
    ret, frame = cap.read()
    
    if not ret:  
        break

    frame_filename = os.path.join(output_dir, f"frame_{frame_count + 1:03d}.jpg")
    cv2.imwrite(frame_filename, frame)

    frame_count += 1

cap.release()
print(f"Saved {frame_count} frames from {start_time_sec} seconds into {output_dir}")
