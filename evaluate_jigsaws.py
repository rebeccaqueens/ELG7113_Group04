from pathlib import Path
from collections import Counter
import csv

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, Subset

from torchvision.models.video import (
    r3d_18,
    R3D_18_Weights
)


# ==================================================
# SETTINGS
# ==================================================

VALIDATION_PARTICIPANT = "B"

BATCH_SIZE = 4

# IMPORTANT:
# The model you already trained had 15 outputs,
# so evaluation must also use 15 outputs.
NUM_CLASSES = 15


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

RESULTS_DIR = (
    ROOT_DIR
    / "evaluation_results"
)

RESULTS_DIR.mkdir(
    exist_ok=True
)


# ==================================================
# CHECK MODEL
# ==================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Best model not found:\n{MODEL_PATH}"
    )


# ==================================================
# PREPROCESSING
# ==================================================

weights = R3D_18_Weights.DEFAULT

preprocess = weights.transforms()


# ==================================================
# GET PARTICIPANT
# ==================================================

def get_participant(trial_name):

    # Suturing_B001 -> B
    code = trial_name.split("_")[-1]

    return code[0]


# ==================================================
# GET VIDEO CLIP
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

        # JIGSAWS uses 1-based frame numbers.
        # OpenCV uses 0-based frame numbers.

        video.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number - 1
        )

        success, frame = video.read()

        if not success:

            video.release()

            raise RuntimeError(
                f"Could not read frame "
                f"{frame_number} from "
                f"{video_path}"
            )


        # OpenCV gives BGR.
        # R3D expects RGB.

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frames.append(frame)


    video.release()


    frames = np.stack(
        frames
    )


    frames = torch.from_numpy(
        frames
    )


    # [T, H, W, C]
    #       ->
    # [T, C, H, W]

    frames = frames.permute(
        0,
        3,
        1,
        2
    )


    return frames


# ==================================================
# DATASET
# ==================================================

class JIGSAWSDataset(Dataset):

    def __init__(self, root):

        self.root = Path(root)

        self.samples = []


        videos = list(
            self.root.rglob(
                "*_capture1.avi"
            )
        )


        print(
            "Videos found:",
            len(videos)
        )


        for video_path in videos:

            trial_name = (
                video_path.name.replace(
                    "_capture1.avi",
                    ""
                )
            )


            participant = (
                get_participant(
                    trial_name
                )
            )


            transcription_files = [

                path

                for path in self.root.rglob(
                    f"{trial_name}.txt"
                )

                if (
                    path.parent.name.lower()
                    == "transcriptions"
                )
            ]


            if len(
                transcription_files
            ) == 0:

                print(
                    "No transcription found:",
                    trial_name
                )

                continue


            transcription_path = (
                transcription_files[0]
            )


            with open(
                transcription_path,
                "r"
            ) as file:

                for line in file:

                    parts = (
                        line.strip().split()
                    )


                    if len(parts) < 3:

                        continue


                    start_frame = int(
                        parts[0]
                    )

                    end_frame = int(
                        parts[1]
                    )

                    gesture = parts[2]


                    if not gesture.startswith(
                        "G"
                    ):

                        continue


                    # Same mapping used during training:
                    #
                    # G1 -> 0
                    # G2 -> 1
                    # ...
                    # G15 -> 14

                    label = (
                        int(
                            gesture[1:]
                        )
                        - 1
                    )


                    self.samples.append(
                        (
                            video_path,
                            start_frame,
                            end_frame,
                            label,
                            gesture,
                            participant,
                            trial_name
                        )
                    )


        print(
            "Gesture segments found:",
            len(self.samples)
        )


    def __len__(self):

        return len(self.samples)


    def __getitem__(
        self,
        index
    ):

        (
            video_path,
            start_frame,
            end_frame,
            label,
            gesture,
            participant,
            trial_name
        ) = self.samples[index]


        clip = get_video_clip(
            video_path,
            start_frame,
            end_frame
        )


        clip = preprocess(
            clip
        )


        return clip, label


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    print()
    print(
        "========================================"
    )
    print(
        "JIGSAWS MODEL EVALUATION"
    )
    print(
        "========================================"
    )


    # ----------------------------------------------
    # DEVICE
    # ----------------------------------------------

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


    # ----------------------------------------------
    # DATASET
    # ----------------------------------------------

    dataset = JIGSAWSDataset(
        JIGSAWS_ROOT
    )


    validation_indices = []


    for index, sample in enumerate(
        dataset.samples
    ):

        participant = sample[5]

        if (
            participant
            == VALIDATION_PARTICIPANT
        ):

            validation_indices.append(
                index
            )


    validation_dataset = Subset(
        dataset,
        validation_indices
    )


    print(
        "\nValidation participant:",
        VALIDATION_PARTICIPANT
    )

    print(
        "Validation samples:",
        len(validation_dataset)
    )


    if len(
        validation_dataset
    ) == 0:

        raise RuntimeError(
            "No validation samples found."
        )


    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )


    # ----------------------------------------------
    # MODEL
    # ----------------------------------------------

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
        "\nBest model loaded:"
    )

    print(
        MODEL_PATH
    )


    # ----------------------------------------------
    # EVALUATION
    # ----------------------------------------------

    all_labels = []

    all_predictions = []


    print(
        "\nRunning validation..."
    )


    with torch.no_grad():

        for batch_number, (
            clips,
            labels
        ) in enumerate(
            validation_loader
        ):

            clips = clips.to(
                device
            )

            labels = labels.to(
                device
            )


            outputs = model(
                clips
            )


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            all_labels.extend(
                labels.cpu().tolist()
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )


            if (
                batch_number + 1
            ) % 5 == 0:

                print(
                    f"Validation batch "
                    f"{batch_number + 1}"
                    f"/{len(validation_loader)}"
                )


    # ----------------------------------------------
    # OVERALL ACCURACY
    # ----------------------------------------------

    correct = sum(

        predicted == actual

        for predicted, actual
        in zip(
            all_predictions,
            all_labels
        )
    )


    overall_accuracy = (
        100
        * correct
        / len(all_labels)
    )


    print()
    print(
        "========================================"
    )

    print(
        "OVERALL RESULT"
    )

    print(
        "========================================"
    )

    print(
        f"Correct: "
        f"{correct}/{len(all_labels)}"
    )

    print(
        f"Validation accuracy: "
        f"{overall_accuracy:.2f}%"
    )


    # ----------------------------------------------
    # PER-GESTURE ACCURACY
    # ----------------------------------------------

    gesture_total = Counter()

    gesture_correct = Counter()


    for actual, predicted in zip(
        all_labels,
        all_predictions
    ):

        gesture_name = (
            f"G{actual + 1}"
        )

        gesture_total[
            gesture_name
        ] += 1


        if actual == predicted:

            gesture_correct[
                gesture_name
            ] += 1


    print()
    print(
        "========================================"
    )

    print(
        "PER-GESTURE ACCURACY"
    )

    print(
        "========================================"
    )


    per_gesture_rows = []


    gestures_present = sorted(
        gesture_total.keys(),
        key=lambda name: int(
            name[1:]
        )
    )


    for gesture in gestures_present:

        total = gesture_total[
            gesture
        ]

        correct_for_gesture = (
            gesture_correct[
                gesture
            ]
        )

        accuracy = (
            100
            * correct_for_gesture
            / total
        )


        print(
            f"{gesture:>4} : "
            f"{correct_for_gesture:>3}"
            f"/{total:<3} "
            f"({accuracy:6.2f}%)"
        )


        per_gesture_rows.append(
            (
                gesture,
                correct_for_gesture,
                total,
                accuracy
            )
        )


    # ----------------------------------------------
    # SAVE PER-GESTURE CSV
    # ----------------------------------------------

    per_gesture_path = (
        RESULTS_DIR
        / "per_gesture_accuracy.csv"
    )


    with open(
        per_gesture_path,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(
            file
        )


        writer.writerow(
            [
                "Gesture",
                "Correct",
                "Total",
                "Accuracy_Percent"
            ]
        )


        writer.writerows(
            per_gesture_rows
        )


    # ----------------------------------------------
    # CONFUSION MATRIX
    # ----------------------------------------------

    confusion_matrix = np.zeros(
        (
            NUM_CLASSES,
            NUM_CLASSES
        ),
        dtype=int
    )


    for actual, predicted in zip(
        all_labels,
        all_predictions
    ):

        confusion_matrix[
            actual,
            predicted
        ] += 1


    confusion_csv_path = (
        RESULTS_DIR
        / "confusion_matrix.csv"
    )


    np.savetxt(
        confusion_csv_path,
        confusion_matrix,
        delimiter=",",
        fmt="%d"
    )


    # ----------------------------------------------
    # CONFUSION MATRIX IMAGE
    # ----------------------------------------------

    labels = [
        f"G{i}"
        for i in range(
            1,
            NUM_CLASSES + 1
        )
    ]


    plt.figure(
        figsize=(12, 10)
    )


    plt.imshow(
        confusion_matrix
    )


    plt.title(
        "JIGSAWS Gesture Confusion Matrix"
    )


    plt.xlabel(
        "Predicted Gesture"
    )


    plt.ylabel(
        "Actual Gesture"
    )


    plt.xticks(
        range(NUM_CLASSES),
        labels,
        rotation=45
    )


    plt.yticks(
        range(NUM_CLASSES),
        labels
    )


    plt.colorbar()


    for actual in range(
        NUM_CLASSES
    ):

        for predicted in range(
            NUM_CLASSES
        ):

            value = confusion_matrix[
                actual,
                predicted
            ]


            if value > 0:

                plt.text(
                    predicted,
                    actual,
                    str(value),
                    ha="center",
                    va="center"
                )


    plt.tight_layout()


    confusion_png_path = (
        RESULTS_DIR
        / "confusion_matrix.png"
    )


    plt.savefig(
        confusion_png_path,
        dpi=200
    )


    plt.close()


    # ----------------------------------------------
    # PREDICTION DISTRIBUTION
    # ----------------------------------------------

    prediction_counts = Counter(
        all_predictions
    )


    print()
    print(
        "========================================"
    )

    print(
        "PREDICTION COUNTS"
    )

    print(
        "========================================"
    )


    for prediction in sorted(
        prediction_counts
    ):

        print(
            f"G{prediction + 1}: "
            f"{prediction_counts[prediction]}"
        )


    # ----------------------------------------------
    # FINISHED
    # ----------------------------------------------

    print()
    print(
        "========================================"
    )

    print(
        "EVALUATION COMPLETE"
    )

    print(
        "========================================"
    )


    print(
        "\nResults saved to:"
    )

    print(
        RESULTS_DIR
    )


    print(
        "\nPer-gesture accuracy:"
    )

    print(
        per_gesture_path
    )


    print(
        "\nConfusion matrix CSV:"
    )

    print(
        confusion_csv_path
    )


    print(
        "\nConfusion matrix image:"
    )

    print(
        confusion_png_path
    )