import cv2
import numpy as np
import torch
import os
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

video_path = r"C:\Users\happy\OneDrive\Documents\Robotic Surgery\JIGSAWS\Suturing\Suturing\video\Suturing_B001_capture1.avi"


transcription_path = r"C:\Users\happy\OneDrive\Documents\Robotic Surgery\JIGSAWS\Suturing\Suturing\transcriptions\Suturing_B001.txt"


def get_video_clip(video_path, start_frame, end_frame, num_frames=16):
    video = cv2.VideoCapture(video_path)

    frame_numbers = np.linspace(
        start_frame,
        end_frame,
        num_frames,
        dtype=int
    )

    frames = []

    for frame_number in frame_numbers:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        success, frame = video.read()

        if not success:
            print("Could not read frame:", frame_number)
            continue

        # OpenCV gives BGR, convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frames.append(frame)

    video.release()

    frames = np.stack(frames)

    # Current shape:
    # [frames, height, width, channels]

    frames = torch.from_numpy(frames)

    # Change to:
    # [frames, channels, height, width]

    frames = frames.permute(0, 3, 1, 2)

    return frames


print("Exists:", os.path.exists(video_path))
print("Path:", video_path)

video = cv2.VideoCapture(video_path)

print("Opened:", video.isOpened())


if not video.isOpened():
    print("❌ Could not open video")
    exit()

fps = video.get(cv2.CAP_PROP_FPS)
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))


print("✅ VIDEO OPENED")
print("FPS:", fps)
print("Number of frames:", frame_count)
print("Resolution:", width, "x", height)


segments = []

with open(transcription_path, "r") as file:

    for line in file:

        parts = line.strip().split()

        if len(parts) != 3:
            continue

        start_frame = int(parts[0])
        end_frame = int(parts[1])
        gesture = parts[2]

        segments.append(
            (start_frame, end_frame, gesture)
        )


print("\n✅ TRANSCRIPTION OPENED")
print(
    "Number of gesture segments:",
    len(segments)
)

print("\nFirst 10 gesture segments:")

for segment in segments[:10]:
    print(segment)


start_frame, end_frame, gesture = segments[0]

clip = get_video_clip(
    video_path,
    start_frame,
    end_frame
)


print("\n✅ CLIP CREATED")
print("Gesture:", gesture)
print("Start frame:", start_frame)
print("End frame:", end_frame)
print("Clip shape:", clip.shape)
print("Data type:", clip.dtype)

weights = R3D_18_Weights.DEFAULT

preprocess = weights.transforms()

processed_clip = preprocess(clip)

print("\n✅ CLIP PREPROCESSED")
print("Before:", clip.shape)
print("After:", processed_clip.shape)

video.release()

model = r3d_18(weights=weights)

print("\n✅ R3D-18 LOADED")


model.fc = nn.Linear(
    model.fc.in_features,
    15
)

print("New output layer:")
print(model.fc)


batch = processed_clip.unsqueeze(0)

print("\nBatch shape:", batch.shape)

model.eval()

with torch.no_grad():
    output = model(batch)

print("Model output shape:", output.shape)

print("New output layer:")
print(model.fc)

# Add a batch dimension
batch = processed_clip.unsqueeze(0)

print("\n✅ BATCH CREATED")
print("Batch shape:", batch.shape)

# Put model into evaluation mode for this test
model.eval()

# We are NOT training yet, so don't calculate gradients
with torch.no_grad():
    output = model(batch)

print("\n✅ VIDEO PASSED THROUGH R3D-18")
print("Output shape:", output.shape)
print("Raw output:")
print(output)