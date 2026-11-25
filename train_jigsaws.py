from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, Subset

from torchvision.models.video import (
    r3d_18,
    R3D_18_Weights
)


# ==================================================
# SETTINGS
# ==================================================

BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 0.001

VALIDATION_PARTICIPANT = "B"

NUM_CLASSES = 15


# ==================================================
# PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent

ROOT_DIR = PROJECT_DIR.parent



JIGSAWS_ROOT = PROJECT_DIR #/ "video"

print("Looking for training data in:", JIGSAWS_ROOT)
print("Folder exists:", JIGSAWS_ROOT.is_dir())
print("Capture 1 videos:", len(list(JIGSAWS_ROOT.rglob("*_capture1.avi"))))
print("Capture 2 videos:", len(list(JIGSAWS_ROOT.rglob("*_capture2.avi"))))

MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(
    exist_ok=True
)


BEST_MODEL_PATH = (
    MODEL_DIR /
    "jigsaws_r3d18_best.pth"
)




# ==================================================
# R3D PREPROCESS
# ==================================================

weights = R3D_18_Weights.DEFAULT

preprocess = weights.transforms()


# ==================================================
# GET PARTICIPANT
# ==================================================

def get_participant(trial_name):

    # Suturing_B001
    #        |
    #        B

    code = trial_name.split("_")[-1]

    return code[0]


# ==================================================
# VIDEO CLIP LOADER
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


    frame_numbers = np.linspace(
        start_frame,
        end_frame,
        num_frames,
        dtype=int
    )


    frames = []


    for frame_number in frame_numbers:


        video.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number - 1
        )


        success, frame = video.read()


        if not success:

            video.release()

            raise RuntimeError(
                "Could not read frame"
            )


        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        frames.append(frame)



    video.release()


    frames = np.stack(frames)


    frames = torch.from_numpy(
        frames
    )


    # T,H,W,C
    # ->
    # T,C,H,W

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

                if path.parent.name.lower()
                ==
                "transcriptions"

            ]



            if len(
                transcription_files
            ) == 0:

                continue



            transcription_path = (
                transcription_files[0]
            )



            with open(
                transcription_path
            ) as file:



                for line in file:


                    parts = (
                        line.strip()
                        .split()
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
# TRAINING
# ==================================================

def train_epoch(
        model,
        loader,
        optimizer,
        criterion,
        device
):


    model.train()


    correct = 0
    total = 0
    loss_total = 0



    for batch_number, (clips, labels) in enumerate(loader):


        clips = clips.to(device)

        labels = labels.to(device)



        optimizer.zero_grad()



        outputs = model(
            clips
        )


        loss = criterion(
            outputs,
            labels
        )



        loss.backward()


        optimizer.step()
        if (batch_number + 1) % 10 == 0:
            print(f"Training batch {batch_number + 1}/{len(loader)}")



        loss_total += loss.item()



        predictions = (
            torch.argmax(
                outputs,
                dim=1
            )
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)



    return (
        loss_total / len(loader),
        100 * correct / total
    )



# ==================================================
# VALIDATION
# ==================================================

def validate(
        model,
        loader,
        criterion,
        device
):


    model.eval()


    correct = 0
    total = 0
    loss_total = 0



    with torch.no_grad():


        for clips, labels in loader:


            clips = clips.to(device)

            labels = labels.to(device)



            outputs = model(
                clips
            )


            loss = criterion(
                outputs,
                labels
            )


            loss_total += loss.item()



            predictions = torch.argmax(
                outputs,
                dim=1
            )



            correct += (
                predictions == labels
            ).sum().item()



            total += labels.size(0)



    return (
        loss_total / len(loader),
        100 * correct / total
    )



# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        "Using device:",
        device
    )


    dataset = JIGSAWSDataset(
        JIGSAWS_ROOT
    )



    # -----------------------------
    # DATASET CHECK
    # -----------------------------


    participants = Counter()

    gestures = Counter()



    for sample in dataset.samples:

        participants[
            sample[5]
        ] += 1


        gestures[
            sample[4]
        ] += 1



    print("\nPARTICIPANTS")

    for p,c in participants.items():

        print(
            p,
            ":",
            c
        )


    print("\nGESTURES")

    for g,c in gestures.items():

        print(
            g,
            ":",
            c
        )



    # -----------------------------
    # SPLIT
    # -----------------------------


    train_indices = []

    val_indices = []



    for i,sample in enumerate(
        dataset.samples
    ):


        if sample[5] == VALIDATION_PARTICIPANT:

            val_indices.append(i)

        else:

            train_indices.append(i)



    train_dataset = Subset(
        dataset,
        train_indices
    )


    val_dataset = Subset(
        dataset,
        val_indices
    )



    print(
        "\nTraining samples:",
        len(train_dataset)
    )


    print(
        "Validation samples:",
        len(val_dataset)
    )



    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE
    )



    # -----------------------------
    # MODEL
    # -----------------------------


    model = r3d_18(
        weights=weights
    )


    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )



    for p in model.parameters():

        p.requires_grad = False



    for p in model.fc.parameters():

        p.requires_grad = True



    model.to(device)



    criterion = nn.CrossEntropyLoss()



    optimizer = torch.optim.Adam(
        model.fc.parameters(),
        lr=LEARNING_RATE
    )



    best_accuracy = 0



    # -----------------------------
    # TRAIN
    # -----------------------------


    for epoch in range(
        EPOCHS
    ):


        print(
            f"\nEpoch {epoch+1}/{EPOCHS}"
        )


        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )


        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            device
        )



        print(
            "Train loss:",
            train_loss
        )


        print(
            "Train accuracy:",
            train_acc
        )


        print(
            "Validation loss:",
            val_loss
        )


        print(
            "Validation accuracy:",
            val_acc
        )



        if val_acc > best_accuracy:


            best_accuracy = val_acc


            torch.save(
                model.state_dict(),
                BEST_MODEL_PATH
            )


            print(
                "BEST MODEL SAVED"
            )



    print(
        "\nTraining complete"
    )


    print(
        "Best validation accuracy:",
        best_accuracy
    )
