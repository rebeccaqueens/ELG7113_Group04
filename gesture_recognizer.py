from contextlib import ExitStack
from pathlib import Path
from collections import deque
import csv

import cv2
import numpy as np
import torch
import torch.nn as nn
import time


from torchvision.models.video import (
    r3d_18,
    R3D_18_Weights
)


# ==================================================
# SETTINGS
# ==================================================

# Use held-out participant B
TRIAL_NAME = "Suturing_B001"

# Current trained model has 15 output neurons
NUM_CLASSES = 15

# R3D-18 receives 16 sampled frames
MODEL_FRAMES = 16

# Keep the most recent 90 frames
# 90 frames / 30 FPS ≈ 3 seconds
WINDOW_FRAMES = 90

# Make a new prediction every 15 frames
# 15 frames / 30 FPS ≈ every 0.5 seconds
STEP_FRAMES = 15

# Average the last few predictions
# so the displayed gesture does not flicker constantly
SMOOTHING_PREDICTIONS = 5


# Valid gestures observed in the JIGSAWS
# suturing data we loaded
VALID_GESTURES = {
    0,   # G1
    1,   # G2
    2,   # G3
    3,   # G4
    4,   # G5
    5,   # G6
    7,   # G8
    8,   # G9
    9,   # G10
    10   # G11
}


# Set this to False if you want the raw
# 15-class model output with no filtering
FILTER_TO_SUTURING_GESTURES = True


# ==================================================
# PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent

ROOT_DIR = PROJECT_DIR.parent

JIGSAWS_ROOT = ROOT_DIR / "video"

MODEL_PATH = (
    ROOT_DIR
    / "ELG7113_Group04"
    / "models"
    / "jigsaws_r3d18_best.pth"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "ELG7113_Group04"
    / "prediction_results"
)




def iter_predictions():
    """Yield each smoothed video prediction and pause until requested again."""
    with ExitStack() as resources:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        # ==================================================
        # FIND VIDEO
        # ==================================================

        VIDEO_PATH = PROJECT_DIR / "video" / "Suturing_B001_capture1.avi"

        if not VIDEO_PATH.is_file():
            raise FileNotFoundError(f"Video file missing: {VIDEO_PATH}")


        OUTPUT_VIDEO = (
            OUTPUT_DIR
            / f"{TRIAL_NAME}_continuous_prediction.mp4"
        )


        OUTPUT_CSV = (
            OUTPUT_DIR
            / f"{TRIAL_NAME}_continuous_predictions.csv"
        )


        # ==================================================
        # PREPROCESSING
        # ==================================================

        weights = R3D_18_Weights.DEFAULT

        preprocess = weights.transforms()


        # ==================================================
        # DEVICE
        # ==================================================

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )


        print()
        print("========================================")
        print("CONTINUOUS JIGSAWS GESTURE PREDICTION")
        print("========================================")

        print("\nUsing device:", device)


        if device.type == "cuda":

            print(
                "GPU:",
                torch.cuda.get_device_name(0)
            )


        # ==================================================
        # LOAD MODEL
        # ==================================================

        model = r3d_18(
            weights=weights
        )


        model.fc = nn.Linear(
            model.fc.in_features,
            NUM_CLASSES
        )


        state_dict = torch.load(
            MODEL_PATH,
            map_location=device
        )


        model.load_state_dict(
            state_dict
        )


        model = model.to(
            device
        )


        model.eval()


        print("\nModel loaded:")
        print(MODEL_PATH)


        # ==================================================
        # OPEN VIDEO
        # ==================================================

        video = cv2.VideoCapture(
            str(VIDEO_PATH)
        )


        resources.callback(video.release)

        if not video.isOpened():

            raise RuntimeError(
                f"Could not open video:\n{VIDEO_PATH}"
            )


        fps = video.get(
            cv2.CAP_PROP_FPS
        )


        width = int(
            video.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )


        height = int(
            video.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )


        total_frames = int(
            video.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )


        print("\nVideo:")
        print(VIDEO_PATH)

        print("\nFPS:", fps)
        print("Frames:", total_frames)

        print(
            f"Window: {WINDOW_FRAMES} frames "
            f"(~{WINDOW_FRAMES / fps:.2f} seconds)"
        )

        print(
            f"Prediction interval: {STEP_FRAMES} frames "
            f"(~{STEP_FRAMES / fps:.2f} seconds)"
        )


        # ==================================================
        # VIDEO WRITER
        # ==================================================

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )


        writer = cv2.VideoWriter(
            str(OUTPUT_VIDEO),
            fourcc,
            fps,
            (
                width,
                height
            )
        )


        resources.callback(writer.release)

        if not writer.isOpened():

            raise RuntimeError(
                "Could not create output video."
            )


        # ==================================================
        # FRAME BUFFER
        # ==================================================

        frame_buffer = deque(
            maxlen=WINDOW_FRAMES
        )


        # Stores recent probability vectors
        # for smoothing
        prediction_history = deque(
            maxlen=SMOOTHING_PREDICTIONS
        )


        current_prediction = None

        current_confidence = None


        # ==================================================
        # CSV RESULTS
        # ==================================================

        csv_file = open(
            OUTPUT_CSV,
            "w",
            newline=""
        )


        resources.callback(csv_file.close)

        csv_writer = csv.writer(
            csv_file
        )


        csv_writer.writerow(
            [
                "frame",
                "time_seconds",
                "predicted_gesture",
                "confidence_percent"
            ]
        )


        # ==================================================
        # PREDICTION FUNCTION
        # ==================================================

        def predict_window(frames):

            # ----------------------------------------------
            # SAMPLE 16 FRAMES ACROSS CURRENT WINDOW
            # ----------------------------------------------

            indices = np.linspace(
                0,
                len(frames) - 1,
                MODEL_FRAMES,
                dtype=int
            )


            sampled_frames = []


            for index in indices:

                frame = frames[index]

                # OpenCV BGR -> RGB
                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                sampled_frames.append(
                    frame
                )


            sampled_frames = np.stack(
                sampled_frames
            )


            clip = torch.from_numpy(
                sampled_frames
            )


            # [T, H, W, C]
            # ->
            # [T, C, H, W]

            clip = clip.permute(
                0,
                3,
                1,
                2
            )


            clip = preprocess(
                clip
            )


            # Add batch dimension
            #
            # [C,T,H,W]
            # ->
            # [1,C,T,H,W]

            batch = clip.unsqueeze(
                0
            )


            batch = batch.to(
                device
            )


            # ----------------------------------------------
            # MODEL
            # ----------------------------------------------

            with torch.no_grad():

                output = model(
                    batch
                )


                # Optional masking of classes that
                # do not occur in suturing

                if FILTER_TO_SUTURING_GESTURES:

                    for class_index in range(
                        NUM_CLASSES
                    ):

                        if (
                            class_index
                            not in VALID_GESTURES
                        ):

                            output[
                                0,
                                class_index
                            ] = float("-inf")


                probabilities = torch.softmax(
                    output,
                    dim=1
                )


            probabilities = (
                probabilities
                .squeeze(0)
                .cpu()
                .numpy()
            )


            return probabilities


        # ==================================================
        # RUN THROUGH VIDEO
        # ==================================================

        print()
        print("========================================")
        print("PROCESSING VIDEO")
        print("========================================")
        print()


        frame_number = 0

        prediction_count = 0

        video_start = time.monotonic()
        
        while True:

            success, frame = video.read()


            if not success:
                break


            frame_number += 1
            due = video_start + (frame_number - 1) / fps
            remaining = due - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)


            prediction = None

            # Keep current frame
            frame_buffer.append(
                frame.copy()
            )


            # ==================================================
            # MAKE PREDICTION
            # ==================================================

            if (
                len(frame_buffer)
                == WINDOW_FRAMES
                and
                (
                    frame_number
                    - WINDOW_FRAMES
                )
                % STEP_FRAMES
                == 0
            ):

                probabilities = predict_window(
                    list(frame_buffer)
                )


                prediction_history.append(
                    probabilities
                )


                # ----------------------------------------------
                # SMOOTH RECENT PREDICTIONS
                # ----------------------------------------------

                average_probabilities = np.mean(
                    np.stack(
                        prediction_history
                    ),
                    axis=0
                )


                predicted_index = int(
                    np.argmax(
                        average_probabilities
                    )
                )


                current_prediction = (
                    f"G{predicted_index + 1}"
                )


                current_confidence = (
                    float(
                        average_probabilities[
                            predicted_index
                        ]
                    )
                    * 100
                )


                prediction_count += 1
                prediction = (current_prediction, frame_number, frame_number / fps, current_confidence)


                time_seconds = (
                    frame_number
                    / fps
                )


                csv_writer.writerow(
                    [
                        frame_number,
                        f"{time_seconds:.3f}",
                        current_prediction,
                        f"{current_confidence:.2f}"
                    ]
                )


                print(
                    f"Frame {frame_number:5d} | "
                    f"Time {time_seconds:7.2f}s | "
                    f"Prediction: "
                    f"{current_prediction} | "
                    f"Confidence: "
                    f"{current_confidence:5.1f}%"
                )


            # ==================================================
            # DRAW OVERLAY
            # ==================================================

            if current_prediction is None:

                cv2.putText(
                    frame,
                    "Buffering video...",
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )


                cv2.putText(
                    frame,
                    f"Need {WINDOW_FRAMES} frames",
                    (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

            else:

                cv2.putText(
                    frame,
                    f"Predicted Gesture: "
                    f"{current_prediction}",
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )


                cv2.putText(
                    frame,
                    f"Confidence: "
                    f"{current_confidence:.1f}%",
                    (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )


                cv2.putText(
                    frame,
                    "Continuous prediction - no transcription",
                    (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )


            # ==================================================
            # PROGRESS
            # ==================================================

            if frame_number % 300 == 0:

                print(
                    f"Processed "
                    f"{frame_number}/{total_frames} "
                    f"video frames"
                )


            writer.write(
                frame
            )
            if prediction is not None:
                yield prediction

        print(f"Saved {prediction_count} predictions to {OUTPUT_CSV}")
        print(f"Annotated video: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    for _ in iter_predictions():
        pass
