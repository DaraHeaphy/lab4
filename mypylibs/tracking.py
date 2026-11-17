
import cv2 as cv
import numpy as np


def draw_trail(frame, points, colour, thickness=2):
    """Draw a trajectory, skipping frames without valid points."""
    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue

        cv.line(
            frame,
            points[i - 1],
            points[i],
            colour,
            thickness,
            cv.LINE_AA
        )


def run_tracking_video(video_in, video_out, measurements, steps,
                       min_occlusion=5):

    cap = cv.VideoCapture(video_in)

    if not cap.isOpened():
        raise FileNotFoundError(video_in)

    fps = cap.get(cv.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    writer = cv.VideoWriter(
        video_out,
        cv.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    measured_trail = []
    predicted_trail = []
    corrected_trail = []

    # Independent predictions made after each reappearance
    uncorrected_branches = []

    gap = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        m = measurements[frame_idx]
        s = steps[frame_idx]

        # Record measured and filtered positions
        measured_trail.append(
            (int(m.x), int(m.y)) if m.valid else None
        )

        if s.started:
            predicted_trail.append(
                tuple(np.round(s.pred[:2]).astype(int))
            )

            corrected_trail.append(
                tuple(np.round(s.post[:2]).astype(int))
            )
        else:
            predicted_trail.append(None)
            corrected_trail.append(None)

        # Detect when the object reappears after an occlusion
        if s.corrected and gap >= min_occlusion:

            # Start a separate trajectory using the prediction
            # BEFORE the new measurement corrected the filter.
            uncorrected_branches.append({
                'start': frame_idx,
                'x': float(s.pred[0]),
                'y': float(s.pred[1]),
                'vx': float(s.pred[2]),
                'vy': float(s.pred[3])
            })

        if s.corrected:
            gap = 0
        elif s.started:
            gap += 1

        # Draw the four main trajectories
        draw_trail(frame, measured_trail, (0, 0, 255))
        draw_trail(frame, predicted_trail, (0, 255, 255))
        draw_trail(frame, corrected_trail, (0, 255, 0))

        # Draw the uncorrected counterfactual predictions
        for branch in uncorrected_branches:
            elapsed = frame_idx - branch['start']

            x = branch['x'] + branch['vx'] * elapsed
            y = branch['y'] + branch['vy'] * elapsed

            point = (int(round(x)), int(round(y)))

            # Dashed-looking trajectory
            if elapsed % 2 == 0:
                cv.circle(frame, point, 3, (255, 140, 0), -1)

        # Show current filter position
        if s.started:
            p = tuple(np.round(s.post[:2]).astype(int))
            cv.circle(frame, p, 6, (0, 255, 0), -1)

        status = 'MEASUREMENT' if s.corrected else 'PREDICTION'

        cv.putText(
            frame,
            f'Frame {frame_idx}: {status}',
            (20, 35),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
