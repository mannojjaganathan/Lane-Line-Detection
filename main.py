
import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip


def interested_region(img, vertices):
    """Keep only the polygon-shaped road region."""
    mask = np.zeros_like(img)
    fill_value = 255 if img.ndim == 2 else (255,) * img.shape[2]

    cv2.fillPoly(mask, vertices, fill_value)
    return cv2.bitwise_and(img, mask)



def detect_lane_lines(edges):
    height, width = edges.shape[:2]

    # Detect straight-line segments using the Hough transform.
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=35,
        maxLineGap=100
    )

    output = np.zeros((height, width, 3), dtype=np.uint8)

    if lines is None:
        return output

    left_points = []
    right_points = []

    # Ignore the upper portion of the image, where road markings
    # are small and difficult to distinguish.
    y_bottom = height - 1
    y_top = int(height * 0.55)

    for x1, y1, x2, y2 in lines.reshape(-1, 4):

        dx = x2 - x1
        dy = y2 - y1

        # Ignore horizontal or nearly horizontal segments.
        if abs(dy) < 10:
            continue

        # Ignore segments located too far up the image.
        if max(y1, y2) < height * 0.45:
            continue

        # Calculate where the segment would intersect the bottom
        # of the image.
        x_bottom = x1 + (y_bottom - y1) * dx / dy

        # Ignore segments whose projected positions are implausible.
        if not (-0.20 * width <= x_bottom <= 1.20 * width):
            continue

        points = [(x1, y1), (x2, y2)]

        # LEFT ROAD BOUNDARY:
        # Keep candidates well to the left of the image centre.
        # This helps exclude the yellow centre marking.
        if -0.20 * width <= x_bottom < 0.30 * width:
            left_points.extend(points)

        # RIGHT ROAD BOUNDARY:
        # Keep candidates on the right side of the image.
        elif 0.62 * width < x_bottom <= 1.20 * width:
            right_points.extend(points)

    def fit_and_draw(points):
        if len(points) < 4:
            return

        points_array = np.asarray(points, dtype=np.float64)

        try:
            # Fit x as a function of y.
            slope, intercept = np.polyfit(
                points_array[:, 1],
                points_array[:, 0],
                1
            )
        except (ValueError, np.linalg.LinAlgError):
            return

        x_bottom = int(slope * y_bottom + intercept)
        x_top = int(slope * y_top + intercept)

        # Avoid drawing implausible lines.
        if not (-width <= x_bottom <= 2 * width):
            return

        if not (-width <= x_top <= 2 * width):
            return

        # Draw the detected road boundary in red.
        cv2.line(
            output,
            (x_bottom, y_bottom),
            (x_top, y_top),
            (0, 0, 255),
            8,
            cv2.LINE_AA
        )

    fit_and_draw(left_points)
    fit_and_draw(right_points)

    return output



def process_image(image):
    """Process one RGB video frame and return an RGB result."""
    # MoviePy supplies RGB frames.
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Identify white and yellow lane markings.
    white_mask = cv2.inRange(gray, 200, 255)

    yellow_lower = np.array([15, 80, 80], dtype=np.uint8)
    yellow_upper = np.array([40, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    lane_mask = cv2.bitwise_or(white_mask, yellow_mask)

    filtered = cv2.bitwise_and(gray, lane_mask)
    blurred = cv2.GaussianBlur(filtered, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    height, width = edges.shape

    # Focus on the road area in front of the vehicle.
    vertices = np.array(
    [[
        (0, height - 1),
        (int(width * 0.25), int(height * 0.45)),
        (int(width * 0.75), int(height * 0.45)),
        (width - 1, height - 1),
    ]],
    dtype=np.int32,
)

    road_edges = interested_region(edges, vertices)
    cv2.imwrite("debug_edges.png", edges)
    cv2.imwrite("debug_road_edges.png", road_edges)
    lane_overlay = detect_lane_lines(road_edges)

    # Blend the lane markings with the original RGB frame.
    result = cv2.addWeighted(image, 0.85, lane_overlay, 1.0, 0)

    return result


if __name__ == "__main__":
    input_video = "test2.mp4"
    output_video = "output.mp4"

    if not os.path.isfile(input_video):
        raise FileNotFoundError(
            f"Could not find {input_video}. "
            "Place the video in the same folder as main.py."
        )

    print(f"Processing video: {input_video}")

    with VideoFileClip(input_video) as clip:
        processed_clip = clip.fl_image(process_image)

        try:
            processed_clip.write_videofile(
                output_video,
                audio=False,
                codec="libx264",
                fps=clip.fps,
            )
        finally:
            processed_clip.close()

    print(f"Finished! Output saved to: {output_video}")
