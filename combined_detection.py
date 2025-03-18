import cv2
import numpy as np
from ultralytics import YOLO
import statistics

# ----------------------------
# Part 1: Yard Line Detection (unchanged)
# ----------------------------

# Load the image
frame = cv2.imread("./play02/frame_001.jpg")
if frame is None:
    print("Error: Could not load image.")
    exit()

H, W, _ = frame.shape

# Create a mask that retains only the top 1/3 of the image (where yard lines typically are)
mask = np.zeros((H, W), dtype=np.uint8)
mask[0:H//3, :] = 255  # white in the top 1/3
masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

# Preprocess: convert to grayscale, blur, and detect edges
gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)

# Detect line segments with HoughLinesP
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=20, maxLineGap=10)

# Filter for nearly vertical lines (the yard lines, even if rotated, are roughly similar in orientation)
detected_lines = []
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(abs(angle) - 90) < 45:  # near vertical orientation
            detected_lines.append([x1, y1, x2, y2])

# Combine lines that are close together based on their mid-x positions
combined_lines = []
if detected_lines:
    detected_lines.sort(key=lambda l: (l[0] + l[2]) / 2)
    cluster_threshold = 100  # pixels; adjust as needed
    current_cluster = [detected_lines[0]]
    
    for line in detected_lines[1:]:
        mid_current = (line[0] + line[2]) / 2
        cluster_mid = np.mean([(l[0] + l[2]) / 2 for l in current_cluster])
        if abs(mid_current - cluster_mid) <= cluster_threshold:
            current_cluster.append(line)
        else:
            x1_avg = int(np.mean([l[0] for l in current_cluster]))
            y1_avg = int(np.mean([l[1] for l in current_cluster]))
            x2_avg = int(np.mean([l[2] for l in current_cluster]))
            y2_avg = int(np.mean([l[3] for l in current_cluster]))
            combined_lines.append([x1_avg, y1_avg, x2_avg, y2_avg])
            current_cluster = [line]
    if current_cluster:
        x1_avg = int(np.mean([l[0] for l in current_cluster]))
        y1_avg = int(np.mean([l[1] for l in current_cluster]))
        x2_avg = int(np.mean([l[2] for l in current_cluster]))
        y2_avg = int(np.mean([l[3] for l in current_cluster]))
        combined_lines.append([x1_avg, y1_avg, x2_avg, y2_avg])

# Extend each combined line to span the full height of the image
extended_lines = []
for line in combined_lines:
    x1, y1, x2, y2 = line
    if x2 != x1:
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        x1_new = int((0 - intercept) / slope)
        x2_new = int((H - intercept) / slope)
        extended_lines.append([x1_new, 0, x2_new, H])
    else:
        extended_lines.append([x1, 0, x2, H])

# Draw extended yard lines (green) for visualization
output = frame.copy()
for line in extended_lines:
    cv2.line(output, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 4)

# ----------------------------
# Part 2: User Click to Define Ball Position
# ----------------------------

ball_point = None
def click_ball(event, x, y, flags, param):
    global ball_point
    if event == cv2.EVENT_LBUTTONDOWN:
        ball_point = (x, y)
        print("Ball clicked at:", ball_point)

cv2.namedWindow("Output")
cv2.setMouseCallback("Output", click_ball)

while True:
    cv2.imshow("Output", output)
    key = cv2.waitKey(1) & 0xFF
    if ball_point is not None or key == ord('q'):
        break

# ----------------------------
# Part 3: Find Nearest Yard Line and Draw Scrimmage Line
# ----------------------------
if ball_point is None:
    print("No ball was clicked. Exiting.")
    exit()

# Compute perpendicular distance from ball to each extended yard line and choose the nearest
def point_to_line_distance(pt, line):
    # line: [x1, y1, x2, y2]
    x0, y0 = pt
    x1, y1, x2, y2 = line
    return abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

distances = [point_to_line_distance(ball_point, line) for line in extended_lines]
nearest_line = extended_lines[np.argmin(distances)]
print("Nearest yard line:", nearest_line)

# Compute the line equation (in two-point form) for the nearest yard line.
# We'll use its slope to draw the scrimmage line through the ball, parallel to the yard line.
x1_n, y1_n, x2_n, y2_n = nearest_line
if x2_n != x1_n:
    slope = (y2_n - y1_n) / (x2_n - x1_n)
else:
    slope = float('inf')

# To draw a line through the ball parallel to the nearest yard line, we compute its intercept:
if slope != float('inf'):
    intercept_ball = ball_point[1] - slope * ball_point[0]
    # Compute two points along this line at the top and bottom of the image:
    scrim_pt1 = (0, int(intercept_ball))
    scrim_pt2 = (W, int(slope * W + intercept_ball))
else:
    # If the line is vertical, the scrimmage line is vertical at ball_point[0]
    scrim_pt1 = (ball_point[0], 0)
    scrim_pt2 = (ball_point[0], H)

# Draw the scrimmage line in red
cv2.line(output, scrim_pt1, scrim_pt2, (0, 0, 255), 4)

# ----------------------------
# Part 4: YOLO Detection Over the Whole Image
# ----------------------------

model = YOLO('yolov8m.pt')
results = model(frame, conf=0.1)

detections = []
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()  # format: [x1, y1, x2, y2]
    classes = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    for box, cls, conf in zip(boxes, classes, confidences):
        if int(cls) == 0:  # Only keep person detections
            detections.append([box[0], box[1], box[2], box[3], conf])

# ----------------------------
# Part 5: Non-Maximum Suppression (NMS)
# ----------------------------
def nms(boxes, conf_threshold=0.3, nms_threshold=0.4):
    if len(boxes) == 0:
        return []
    boxes_np = np.array(boxes)
    x1 = boxes_np[:, 0]
    y1 = boxes_np[:, 1]
    x2 = boxes_np[:, 2]
    y2 = boxes_np[:, 3]
    scores = boxes_np[:, 4]
    indices = cv2.dnn.NMSBoxes(boxes_np[:, :4].tolist(), scores.tolist(), conf_threshold, nms_threshold)
    return [boxes[i] for i in indices.flatten()]

filtered_detections = nms(detections)

# ----------------------------
# Part 6: Compute Distance from Each Player to the Scrimmage Line in "Yard Lines"
# ----------------------------
# To convert pixel distances to yard lines, we need the average spacing between yard lines.
# Since yard lines are not vertical, compute spacing along the direction perpendicular to the nearest line.
# For the nearest yard line, compute its unit normal.
if x2_n != x1_n:
    n_raw = np.array([- (y2_n - y1_n), x2_n - x1_n])  # normal vector
    n_norm = np.linalg.norm(n_raw)
    unit_normal = n_raw / n_norm
else:
    unit_normal = np.array([1, 0])  # For a vertical line

# Compute each extended yard line's offset by projecting its midpoint onto the normal
offsets = []
for line in extended_lines:
    mid_pt = np.array([(line[0] + line[2]) / 2, (line[1] + line[3]) / 2])
    offset = np.dot(mid_pt, unit_normal)
    offsets.append(offset)
offsets = sorted(offsets)

if len(offsets) > 1:
    spacings = [offsets[i+1] - offsets[i] for i in range(len(offsets)-1)]
    avg_spacing = statistics.median(spacings)
else:
    avg_spacing = 1  # fallback

print("Average yard line spacing (pixels along normal):", avg_spacing)

# For each player detection, compute the perpendicular distance from the player's center to the scrimmage line.
def distance_point_to_line(pt, line_pt1, line_pt2):
    x0, y0 = pt
    x1, y1 = line_pt1
    x2, y2 = line_pt2
    return abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2-x1)**2)

def calculate_signed_distance(player_center, ball_point, scrim_pt1, scrim_pt2):
    """
    Calculate signed distance from player to line of scrimmage, accounting for camera angle
    """
    # Calculate raw distance
    pixel_distance = distance_point_to_line(player_center, scrim_pt1, scrim_pt2)
    
    # Calculate vector from ball to player
    ball_to_player = np.array([player_center[0] - ball_point[0], player_center[1] - ball_point[1]])
    
    # Calculate line direction vector
    line_vector = np.array([scrim_pt2[0] - scrim_pt1[0], scrim_pt2[1] - scrim_pt1[1]])
    line_normal = np.array([-line_vector[1], line_vector[0]])  # Rotate 90 degrees
    
    # Determine sign based on which side of the line the player is on
    sign = 1 if np.dot(ball_to_player, line_normal) > 0 else -1
    
    return pixel_distance * sign

def classify_defensive_position(distance, x_coord, field_width):
    """
    Enhanced position classification with adjusted ranges and confidence calculation
    """
    relative_x = x_coord / field_width
    
    # Adjusted position ranges (in yards)
    positions = {
        'Defensive Line': {'min_dist': -0.5, 'max_dist': 2.0},
        'Linebacker': {'min_dist': 2.0, 'max_dist': 6.0},
        'Safety': {'min_dist': 7.0, 'max_dist': 15.0},
        'Cornerback': {'min_dist': 2.0, 'max_dist': 8.0}
    }
    
    position_confidence = {}
    
    for pos, ranges in positions.items():
        if ranges['min_dist'] <= distance <= ranges['max_dist']:
            # Base confidence calculation
            dist_range = ranges['max_dist'] - ranges['min_dist']
            dist_from_center = abs(distance - (ranges['min_dist'] + ranges['max_dist'])/2)
            base_conf = 1.0 - (dist_from_center / (dist_range/2))
            
            # Position-specific adjustments
            if pos == 'Defensive Line':
                # Higher confidence for D-Line close to scrimmage
                if distance <= 2.0:
                    base_conf *= 1.2
                # Spread across the line
                base_conf *= 0.9  # Generally high confidence if distance is right
                
            elif pos == 'Linebacker':
                # Linebackers in middle of field
                if 0.2 < relative_x < 0.8:
                    base_conf *= 1.1
                # Higher confidence in typical linebacker depths
                if 3.0 <= distance <= 5.0:
                    base_conf *= 1.1
                    
            elif pos == 'Safety':
                # Safeties typically in middle third
                if 0.3 < relative_x < 0.7:
                    base_conf *= 1.2
                # Higher confidence at typical safety depths
                if 8.0 <= distance <= 12.0:
                    base_conf *= 1.1
                    
            elif pos == 'Cornerback':
                # Cornerbacks typically on outside
                if relative_x < 0.15 or relative_x > 0.85:
                    base_conf *= 1.3
                # Typical cornerback depths
                if 4.0 <= distance <= 7.0:
                    base_conf *= 1.1
            
            position_confidence[pos] = min(base_conf, 1.0)  # Cap confidence at 1.0
    
    if position_confidence:
        best_pos = max(position_confidence.items(), key=lambda x: x[1])
        return best_pos
    return ("Unknown", 0.0)

# Update the main detection loop
player_positions = []
for x1, y1, x2, y2, conf in filtered_detections:
    player_center = ((x1 + x2) / 2, (y1 + y2) / 2)
    
    # Calculate signed distance using new function
    pixel_distance = calculate_signed_distance(player_center, ball_point, scrim_pt1, scrim_pt2)
    yard_line_distance = (abs(pixel_distance) / avg_spacing) * 5
    
    # Apply sign based on which side of the line the player is on
    if pixel_distance < 0:
        yard_line_distance *= -1
    
    # Store player data for defensive/offensive determination
    player_positions.append((player_center, yard_line_distance))

# Determine defensive side based on player clustering
avg_distances = [pos[1] for pos in player_positions]
defensive_side = 1 if statistics.mean(avg_distances) > 0 else -1

# Draw detections with updated logic
for (x1, y1, x2, y2, conf), (player_center, yard_line_distance) in zip(filtered_detections, player_positions):
    # Adjust distance sign based on defensive side
    adjusted_distance = yard_line_distance * defensive_side
    
    # Only classify defensive players
    if (defensive_side > 0 and yard_line_distance > 0) or (defensive_side < 0 and yard_line_distance < 0):
        position, pos_conf = classify_defensive_position(abs(adjusted_distance), player_center[0], W)
    else:
        position, pos_conf = "Offense", 0.0
    
    # Color coding
    position_colors = {
        'Defensive Line': (255, 0, 0),    # Blue
        'Linebacker': (0, 255, 0),        # Green
        'Safety': (0, 0, 255),            # Red
        'Cornerback': (255, 255, 0),      # Cyan
        'Offense': (128, 0, 128),         # Purple
        'Unknown': (128, 128, 128)        # Gray
    }
    
    color = position_colors.get(position, (128, 128, 128))
    cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    
    # Labels
    if position != "Offense":
        label = f"{position} ({pos_conf:.2f})"
    else:
        label = "Offense"
    cv2.putText(output, label, (int(x1), int(y1)-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Distance label
    dist_label = f"{abs(yard_line_distance):.1f} yards"
    cv2.putText(output, dist_label, (int(x1), int(y1)-25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# ----------------------------
# Display the Final Output
# ----------------------------
cv2.imshow("Final Output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
