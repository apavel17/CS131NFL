import cv2
import numpy as np

# Load the image (update the filename/path as necessary)
frame = cv2.imread("./play03/frame_001.jpg")
if frame is None:
    print("Error: Could not load image.")
    exit()

# Get image dimensions
H, W, _ = frame.shape

# Create a mask that retains only the top 1/3 of the image (where yard lines typically are)
mask = np.zeros((H, W), dtype=np.uint8)
mask[0:H//3, :] = 255  # white in the top 1/3

# Apply the mask
masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

# Convert the masked image to grayscale and apply Gaussian blur
gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Use HoughLinesP to detect line segments
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=20, maxLineGap=10)

# Filter for nearly vertical lines (assuming yard lines are vertical in your frame)
detected_lines = []
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Accept lines with an angle near ±90° (within 45° tolerance)
        if abs(abs(angle) - 90) < 45:
            detected_lines.append([x1, y1, x2, y2])

# Combine lines that are close together using a simple threshold on their mid-x coordinates
combined_lines = []
if detected_lines:
    # Sort lines by their mid-x value
    detected_lines.sort(key=lambda l: (l[0] + l[2]) / 2)
    cluster_threshold = 100  # pixels; adjust as needed
    current_cluster = [detected_lines[0]]
    
    for line in detected_lines[1:]:
        mid_current = (line[0] + line[2]) / 2
        # Calculate the mean mid-x of the current cluster
        cluster_mid = np.mean([(l[0] + l[2]) / 2 for l in current_cluster])
        if abs(mid_current - cluster_mid) <= cluster_threshold:
            current_cluster.append(line)
        else:
            # Average the lines in the current cluster
            x1_avg = int(np.mean([l[0] for l in current_cluster]))
            y1_avg = int(np.mean([l[1] for l in current_cluster]))
            x2_avg = int(np.mean([l[2] for l in current_cluster]))
            y2_avg = int(np.mean([l[3] for l in current_cluster]))
            combined_lines.append([x1_avg, y1_avg, x2_avg, y2_avg])
            current_cluster = [line]
    # Process the final cluster
    if current_cluster:
        x1_avg = int(np.mean([l[0] for l in current_cluster]))
        y1_avg = int(np.mean([l[1] for l in current_cluster]))
        x2_avg = int(np.mean([l[2] for l in current_cluster]))
        y2_avg = int(np.mean([l[3] for l in current_cluster]))
        combined_lines.append([x1_avg, y1_avg, x2_avg, y2_avg])

# Optionally, print the combined lines for debugging
for cl in combined_lines:
    print("Combined line:", cl)

# Extend each combined line to span the full height of the image
extended_lines = []
for line in combined_lines:
    x1, y1, x2, y2 = line
    if x2 != x1:
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        # Compute new x positions for y=0 and y=H
        x1_new = int((0 - intercept) / slope)
        x2_new = int((H - intercept) / slope)
        extended_lines.append([x1_new, 0, x2_new, H])
    else:
        extended_lines.append([x1, 0, x2, H])

# Draw the extended lines on a copy of the original image
output = frame.copy()
for line in extended_lines:
    x1, y1, x2, y2 = line
    cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 4)  # thickness=4

# Display the results
cv2.imshow("Blurred", blurred)
cv2.imshow("Detected and Combined Yard Lines", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
