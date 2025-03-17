import cv2
import numpy as np

# --------------------------------------------
# PART 1: Let the user pick four points in the image
# --------------------------------------------
points_src = []  # global or outer scope list to store clicked points

def click_event(event, x, y, flags, param):
    global points_src
    if event == cv2.EVENT_LBUTTONDOWN:
        # On left click, record the point
        points_src.append([x, y])
        print(f"Selected point: ({x}, {y})")

def select_points(image):
    # Make a copy so we don't overwrite original
    clone = image.copy()
    cv2.namedWindow("Select 4 Points")
    cv2.setMouseCallback("Select 4 Points", click_event)

    while True:
        cv2.imshow("Select 4 Points", clone)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to cancel
            break
        # Once we have 4 points, break automatically
        if len(points_src) == 4:
            break
    
    cv2.destroyWindow("Select 4 Points")
    return np.array(points_src, dtype=np.float32)

# --------------------------------------------
# PART 2: Define where we want those four points to map in the output
# --------------------------------------------
def main():
    # Load your all-22 frame
    image_path = './play01/frame_001.jpg'
    img = cv2.imread(image_path)

    # Step A: Pick four known reference points (corner or key markers on the field).
    print("Please click on four distinct points in the image (in order: top-left, top-right, "
          "bottom-left, bottom-right, or any consistent sequence). Press ESC when done.")
    src_pts = select_points(img)
    print("Source points:", src_pts)

    # Step B: Hardcode or define the corresponding points in the rectified/top-down space.
    # For instance, let’s say you want a 1200×600 “bird’s-eye” output.
    # The ordering must match the order in which you clicked the four points above.
    dst_pts = np.float32([
        [0,    0],     # top-left  (in your warped image)
        [1200, 0],     # top-right
        [0,    600],   # bottom-left
        [1200, 600]    # bottom-right
    ])

    # Step C: Compute the perspective transform (homography) from the 4 source points to the 4 destination points.
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Step D: Warp the image to the new perspective
    warped = cv2.warpPerspective(img, M, (1200, 600))

    # Display results
    cv2.imshow("Original Image", img)
    cv2.imshow("Warped Image", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
