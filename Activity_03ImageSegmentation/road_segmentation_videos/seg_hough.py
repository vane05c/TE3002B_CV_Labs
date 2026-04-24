import cv2
import numpy as np

input_video = "road_video2.mp4"
output_video = "road_output_intersection.mp4"

# --------------------------------------------------
# ROI for road area
# --------------------------------------------------
def get_roi_mask(shape):
    h, w = shape[:2]
    roi_mask = np.zeros((h, w), dtype=np.uint8)

    polygon = np.array([[
        (int(0.08 * w), h),
        (int(0.38 * w), int(0.58 * h)),
        (int(0.62 * w), int(0.58 * h)),
        (int(0.92 * w), h)
    ]], dtype=np.int32)

    cv2.fillPoly(roi_mask, polygon, 255)
    return roi_mask

# --------------------------------------------------
# Detect Hough lines from grayscale edges
# --------------------------------------------------
def detect_lane_lines(frame):
    h, w = frame.shape[:2]
    roi_mask = get_roi_mask(frame.shape)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection only inside ROI
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.bitwise_and(edges, roi_mask)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=50,
        maxLineGap=30
    )

    left_lines = []
    right_lines = []

    if lines is None:
        return None, None

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x2 == x1:
            continue

        angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # keep only lower-half lines
        if y1 < 0.55 * h or y2 < 0.55 * h:
            continue

        # left boundary candidate
        if -80 < angle < -20:
            left_lines.append((x1, y1, x2, y2, angle, length))

        # right boundary candidate
        elif 20 < angle < 80:
            right_lines.append((x1, y1, x2, y2, angle, length))

    left_best = max(left_lines, key=lambda x: x[5]) if left_lines else None
    right_best = max(right_lines, key=lambda x: x[5]) if right_lines else None

    return left_best, right_best

# --------------------------------------------------
# Extend line to make full road polygon
# --------------------------------------------------
def extend_line(line, y_bottom, y_top):
    if line is None:
        return None

    x1, y1, x2, y2, angle, length = line

    if y2 == y1:
        return None

    # x = a*y + b
    a = (x2 - x1) / (y2 - y1)
    b = x1 - a * y1

    xb = int(a * y_bottom + b)
    xt = int(a * y_top + b)

    return (xb, y_bottom, xt, y_top)

# --------------------------------------------------
# Build polygon from left/right lines
# --------------------------------------------------
def build_line_polygon(frame_shape, left_line, right_line):
    h, w = frame_shape[:2]
    poly_mask = np.zeros((h, w), dtype=np.uint8)

    if left_line is None or right_line is None:
        return poly_mask

    y_bottom = h
    y_top = int(0.58 * h)

    left_ext = extend_line(left_line, y_bottom, y_top)
    right_ext = extend_line(right_line, y_bottom, y_top)

    if left_ext is None or right_ext is None:
        return poly_mask

    lx1, ly1, lx2, ly2 = left_ext
    rx1, ry1, rx2, ry2 = right_ext

    polygon = np.array([[
        (lx1, ly1),
        (lx2, ly2),
        (rx2, ry2),
        (rx1, ry1)
    ]], dtype=np.int32)

    cv2.fillPoly(poly_mask, polygon, 255)
    return poly_mask

# --------------------------------------------------
# Otsu segmentation
# --------------------------------------------------
def segment_road(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, otsu_mask = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    roi_mask = get_roi_mask(frame.shape)
    otsu_mask = cv2.bitwise_and(otsu_mask, roi_mask)

    kernel = np.ones((5, 5), np.uint8)
    otsu_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_CLOSE, kernel)
    otsu_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_OPEN, kernel)

    return otsu_mask

# --------------------------------------------------
# Main
# --------------------------------------------------
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    raise RuntimeError("Could not open video")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1) Hough lines first
    left_line, right_line = detect_lane_lines(frame)

    # 2) Build polygon from lines
    line_polygon = build_line_polygon(frame.shape, left_line, right_line)

    # 3) Otsu segmentation
    seg_mask = segment_road(frame)

    # 4) Intersect both
    if np.count_nonzero(line_polygon) > 0:
        final_mask = cv2.bitwise_and(seg_mask, line_polygon)
    else:
        final_mask = seg_mask.copy()

    # Visualization
    mask_overlay = np.zeros_like(frame)
    mask_overlay[:, :, 1] = final_mask

    result = cv2.addWeighted(frame, 0.75, mask_overlay, 0.30, 0)

    if left_line is not None:
        x1, y1, x2, y2, angle, _ = left_line
        cv2.line(result, (x1, y1), (x2, y2), (255, 0, 0), 4)
        cv2.putText(result, f"{angle:.1f}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if right_line is not None:
        x1, y1, x2, y2, angle, _ = right_line
        cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 4)
        cv2.putText(result, f"{angle:.1f}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    writer.write(result)
    cv2.imshow("Segmentation + Hough Intersection", result)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()