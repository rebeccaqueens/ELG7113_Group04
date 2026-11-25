from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from torchvision.models.video import (
    r3d_18,
    R3D_18_Weights
)


# ==================================================
# SETTINGS
# ==================================================

# Use a participant-B trial because B was held out
# from training.
TRIAL_NAME = "Suturing_B001"

NUM_CLASSES = 15

NUM_FRAMES = 16


# ==================================================
# PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent

ROOT_DIR = PROJECT_DIR.parent

JIGSAWS_ROOT = ROOT_DIR / "JIGSAWS"

MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "jigsaws_r3d18_best.pth"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "prediction_results"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)


# ==================================================
# PREPROCESSING
# ==================================================

weights = R3D_18_Weights.DEFAULT

preprocess = weights.transforms()


# ==================================================
# FIND VIDEO + TRANSCRIPTION
# ==================================================

video_files = list(
    JIGSAWS_ROOT.rglob(
        f"{TRIAL_NAME}_capture1.avi"
    )
)

transcription_files = [
    path
    for path in JIGSAWS_ROOT.rglob(
        f"{TRIAL_NAME}.txt"
    )
    if path.parent.name.lower()
    == "transcriptions"
]


if len(video_files) == 0:

    raise FileNotFoundError(
        f"Could not find video for "
        f"{TRIAL_NAME}"
    )


if len(transcription_files) == 0:

    raise FileNotFoundError(
        f"Could not find transcription for "
        f"{TRIAL_NAME}"
    )


VIDEO_PATH = video_files[0]

TRANSCRIPTION_PATH = (
    transcription_files[0]
)


print(
    "\nVideo:"
)

print(
    VIDEO_PATH
)

print(
    "\nTranscription:"
)

print(
    TRANSCRIPTION_PATH
)


# ==================================================
# READ TRANSCRIPTION
# ==================================================

segments = []


with open(
    TRANSCRIPTION_PATH,
    "r"
) as file:

    for line in file:

        parts = line.strip().split()

        if len(parts) < 3:
            continue

        start_frame = int(
            parts[0]
        )

        end_frame = int(
            parts[1]
        )

        gesture = parts[2]

        if not gesture.startswith("G"):
            continue

        segments.append(
            {
                "start": start_frame,
                "end": end_frame,
                "actual": gesture
            }
        )


print(
    "\nGesture segments:",
    len(segments)
)


# ==================================================
# GET ONE 16-FRAME CLIP
# ==================================================

def get_video_clip(
    video_path,
    start_frame,
    end_frame,
    num_frames=16
):

    video = cv2.VideoCapture(
        str(video_path)
    )

    if not video.isOpened():

        raise RuntimeError(
            f"Could not open video:\n"
            f"{video_path}"
        )


    frame_numbers = np.linspace(
        start_frame,
        end_frame,
        num_frames,
        dtype=int
    )


    frames = []


    for frame_number in frame_numbers:

        # JIGSAWS frame numbers are 1-based.
        # OpenCV is 0-based.

        video.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number - 1
        )


        success, frame = video.read()


        if not success:

            video.release()

            raise RuntimeError(
                f"Could not read frame "
                f"{frame_number}"
            )


        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        frames.append(
            frame
        )


    video.release()


    frames = np.stack(
        frames
    )


    frames = torch.from_numpy(
        frames
    )


    # [T, H, W, C]
    # ->
    # [T, C, H, W]

    frames = frames.permute(
        0,
        3,
        1,
        2
    )


    return frames


# ==================================================
# DEVICE
# ==================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "\nUsing device:",
    device
)


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


print(
    "\nModel loaded:"
)

print(
    MODEL_PATH
)


# ==================================================
# PREDICT EACH SEGMENT
# ==================================================

print(
    "\n========================================"
)

print(
    "PREDICTING GESTURES"
)

print(
    "========================================\n"
)


correct = 0


with torch.no_grad():

    for segment_number, segment in enumerate(
        segments,
        start=1
    ):

        clip = get_video_clip(
            VIDEO_PATH,
            segment["start"],
            segment["end"],
            NUM_FRAMES
        )


        clip = preprocess(
            clip
        )


        # [C,T,H,W]
        # ->
        # [1,C,T,H,W]

        batch = clip.unsqueeze(
            0
        )


        batch = batch.to(
            device
        )


        output = model(
            batch
        )


        probabilities = torch.softmax(
            output,
            dim=1
        )


        confidence, prediction = (
            torch.max(
                probabilities,
                dim=1
            )
        )


        predicted_label = (
            f"G{prediction.item() + 1}"
        )


        confidence_percent = (
            confidence.item()
            * 100
        )


        actual_label = (
            segment["actual"]
        )


        is_correct = (
            predicted_label
            == actual_label
        )


        if is_correct:

            correct += 1


        segment[
            "prediction"
        ] = predicted_label


        segment[
            "confidence"
        ] = confidence_percent


        segment[
            "correct"
        ] = is_correct


        status = (
            "CORRECT"
            if is_correct
            else "WRONG"
        )


        print(
            f"Segment {segment_number:02d} | "
            f"Frames "
            f"{segment['start']}-"
            f"{segment['end']} | "
            f"Actual: {actual_label} | "
            f"Predicted: {predicted_label} | "
            f"Confidence: "
            f"{confidence_percent:.1f}% | "
            f"{status}"
        )


# ==================================================
# TRIAL ACCURACY
# ==================================================

trial_accuracy = (
    100
    * correct
    / len(segments)
)


print(
    "\n========================================"
)

print(
    "TRIAL RESULT"
)

print(
    "========================================"
)


print(
    f"Correct segments: "
    f"{correct}/{len(segments)}"
)


print(
    f"Trial accuracy: "
    f"{trial_accuracy:.2f}%"
)


# ==================================================
# CREATE ANNOTATED VIDEO
# ==================================================

print(
    "\nCreating annotated video..."
)


video = cv2.VideoCapture(
    str(VIDEO_PATH)
)


if not video.isOpened():

    raise RuntimeError(
        "Could not reopen video."
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


OUTPUT_VIDEO = (
    OUTPUT_DIR
    / f"{TRIAL_NAME}_predicted.mp4"
)


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


frame_number = 1

segment_index = 0


while True:

    success, frame = video.read()


    if not success:
        break


    # Move to correct gesture segment

    while (
        segment_index
        < len(segments) - 1
        and frame_number
        > segments[
            segment_index
        ]["end"]
    ):

        segment_index += 1


    segment = segments[
        segment_index
    ]


    if (
        segment["start"]
        <= frame_number
        <= segment["end"]
    ):

        predicted = segment[
            "prediction"
        ]

        actual = segment[
            "actual"
        ]

        confidence = segment[
            "confidence"
        ]


        prediction_text = (
            f"Predicted: "
            f"{predicted}"
        )


        actual_text = (
            f"Actual: "
            f"{actual}"
        )


        confidence_text = (
            f"Confidence: "
            f"{confidence:.1f}%"
        )


        cv2.putText(
            frame,
            prediction_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        cv2.putText(
            frame,
            actual_text,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        cv2.putText(
            frame,
            confidence_text,
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


    writer.write(
        frame
    )


    frame_number += 1


video.release()

writer.release()


print(
    "\n========================================"
)

print(
    "VIDEO COMPLETE"
)

print(
    "========================================"
)


print(
    "\nAnnotated video saved to:"
)

print(
    OUTPUT_VIDEO
)