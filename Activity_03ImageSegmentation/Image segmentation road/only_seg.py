import cv2
import numpy as np

input_video = "road_video2.mp4"
output_video = "road_segmentation_only.mp4"

def segment_road(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h, w = mask.shape
    roi = np.zeros_like(mask)

    polygon = np.array([[
        (int(0.05 * w), h),
        (int(0.35 * w), int(0.55 * h)),
        (int(0.65 * w), int(0.55 * h)),
        (int(0.95 * w), h)
    ]], dtype=np.int32)

    cv2.fillPoly(roi, polygon, 255)
    mask = cv2.bitwise_and(mask, roi)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask

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

    road_mask = segment_road(frame)

    mask_overlay = np.zeros_like(frame)
    mask_overlay[:, :, 1] = road_mask

    result = cv2.addWeighted(frame, 0.7, mask_overlay, 0.25, 0)

    writer.write(result)
    cv2.imshow("Segmentation Only", result)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()