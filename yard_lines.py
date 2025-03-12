import cv2
import numpy as np

# Load the image (update the filename/path if necessary)
frame = cv2.imread("./play01/frame_001.jpg")
if frame is None:
    print("Error: Could not load image.")
    exit()

# Get image dimensions
H, W, _ = frame.shape

# Create a mask that retains only the top 1/3 of the image
mask = np.zeros((H, W), dtype=np.uint8)
mask[0:H//3, :] = 255  # White in top 1/3, black elsewhere

# Apply the mask to the image
masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

# Convert the masked image to grayscale
gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Use HoughLinesP to detect lines in the edge image
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=20, maxLineGap=10)

# Create a copy of the original frame for drawing detected lines
output = frame.copy()

detected_lines = []

# Filter lines for nearly vertical lines (yard lines)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calculate the angle in degrees relative to horizontal
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Adjust the filter: if yard lines are vertical, they should be near 90 or -90 degrees.
        if abs(abs(angle) - 90) < 45:  # within ±45° of vertical
            detected_lines.append([x1, y1, x2, y2])
            
combined_lines = []
if detected_lines:
    # Sort lines by their mid-x value
    detected_lines.sort(key=lambda l: (l[0] + l[2]) / 2)
    cluster_threshold = 10  # pixels; adjust based on your image resolution
    current_cluster = [detected_lines[0]]
    
    for line in detected_lines[1:]:
        mid_current = (line[0] + line[2]) / 2
        cluster_mid = np.mean([(l[0] + l[2]) / 2 for l in current_cluster])
        if abs(mid_current - cluster_mid) < cluster_threshold:
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

# Draw the combined lines onto the original image with a thicker line width
output = frame.copy()
for line in combined_lines:
    print(line)
    x1, y1, x2, y2 = line
    cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 4)  # thickness=4

# Display the results
cv2.imshow("Blurred", blurred)
cv2.imshow("Detected Yard Lines", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
