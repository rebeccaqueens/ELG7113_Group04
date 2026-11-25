from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


# -------------------------------------------------------------------------
# FOLDERS
# -------------------------------------------------------------------------

base = Path(
    r"C:\Users\rrebe\OneDrive - University of Ottawa\ELG 7113"
)

transcription_folder = base / "transcriptions"
kinematics_folder = base / "kinematics" / "AllGestures"

output_folder = base / "averaged_gestures"
output_folder.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------------
# STORAGE FOR ALL GESTURE EXAMPLES
# -------------------------------------------------------------------------

gesture_examples = {
    f"G{gesture_number}": []
    for gesture_number in range(1, 12)
}


# -------------------------------------------------------------------------
# CONVERT ONE OCCURRENCE INTO RELATIVE MOTION
# -------------------------------------------------------------------------

def make_relative_gesture(slave_arm):
    """
    Convert one N x 19 single-arm trajectory into motion
    relative to its first position and orientation.
    """

    xyz = slave_arm[:, 0:3]

    rotation_matrices = slave_arm[
        :, 3:12
    ].reshape(-1, 3, 3)

    translational_velocity = slave_arm[:, 12:15]
    rotational_velocity = slave_arm[:, 15:18]
    gripper_angle = slave_arm[:, 18:19]

    start_position = xyz[0]
    start_rotation = rotation_matrices[0]

    # Position relative to the first gesture frame, represented
    # in the gesture's initial tool coordinate system.
    relative_xyz = (
        xyz - start_position
    ) @ start_rotation

    # Orientation relative to the first gesture frame.
    relative_rotations = (
        start_rotation.T
        @ rotation_matrices
    )

    # Express velocity vectors in the same relative coordinate frame.
    relative_translational_velocity = (
        translational_velocity @ start_rotation
    )

    relative_rotational_velocity = (
        rotational_velocity @ start_rotation
    )

    return {
        "xyz": relative_xyz,
        "rotations": relative_rotations,
        "trans_vel": relative_translational_velocity,
        "rot_vel": relative_rotational_velocity,
        "gripper": gripper_angle
    }


# -------------------------------------------------------------------------
# READ EVERY AVAILABLE B001–I005 FILE
# -------------------------------------------------------------------------

files_processed = 0

for surgeon in "BCDEFGHI":
    for trial in range(1, 6):

        filename = (
            f"Suturing_{surgeon}{trial:03d}.txt"
        )

        transcription_file = (
            transcription_folder / filename
        )

        kinematics_file = (
            kinematics_folder / filename
        )

        if not transcription_file.is_file():
            print(
                f"Skipping {filename}: "
                "transcription file is missing."
            )
            continue

        if not kinematics_file.is_file():
            print(
                f"Skipping {filename}: "
                "kinematics file is missing."
            )
            continue

        transcriptions = np.loadtxt(
            transcription_file,
            dtype=str,
            ndmin=2
        )

        kinematics = np.loadtxt(
            kinematics_file,
            ndmin=2
        )

        if transcriptions.shape[1] != 3:
            raise ValueError(
                f"{filename}: expected three "
                "transcription columns."
            )

        if kinematics.shape[1] != 76:
            raise ValueError(
                f"{filename}: expected 76 "
                "kinematics columns."
            )

        files_processed += 1

        for start_text, end_text, gesture_name in transcriptions:

            if gesture_name not in gesture_examples:
                continue

            start_frame = int(start_text)
            end_frame = int(end_text)

            if not (
                1 <= start_frame
                <= end_frame
                <= len(kinematics)
            ):
                raise ValueError(
                    f"{filename}, {gesture_name}: "
                    f"invalid frames "
                    f"{start_frame}-{end_frame}."
                )

            # JIGSAWS frames are one-based and inclusive.
            #
            # Original columns 39–57 correspond to the
            # slave-left arm. Python indices are 38:57.
            slave_left = kinematics[
                start_frame - 1:end_frame,
                38:57
            ]

            # Original columns 58–76: slave right (Python 57:76).
            slave_right = kinematics[start_frame - 1:end_frame, 57:76]
            gesture_examples[gesture_name].append({
                "left": make_relative_gesture(slave_left),
                "right": make_relative_gesture(slave_right),
            })

        print(f"Processed {filename}")


print(f"\nFiles processed: {files_processed}")


# -------------------------------------------------------------------------
# INTERPOLATION FUNCTIONS
# -------------------------------------------------------------------------

def interpolate_columns(values, old_time, new_time):
    """Interpolate each numeric column independently."""

    result = np.zeros(
        (len(new_time), values.shape[1])
    )

    for column in range(values.shape[1]):
        result[:, column] = np.interp(
            new_time,
            old_time,
            values[:, column]
        )

    return result


def resample_gesture(gesture, number_of_frames):
    """Resample one gesture to the required number of frames."""

    original_length = len(gesture["xyz"])

    if original_length == 1:
        return {
            key: np.repeat(values, number_of_frames, axis=0)
            for key, values in gesture.items()
        }

    old_time = np.linspace(
        0.0,
        1.0,
        original_length
    )

    new_time = np.linspace(
        0.0,
        1.0,
        number_of_frames
    )

    xyz = interpolate_columns(
        gesture["xyz"],
        old_time,
        new_time
    )

    trans_vel = interpolate_columns(
        gesture["trans_vel"],
        old_time,
        new_time
    )

    rot_vel = interpolate_columns(
        gesture["rot_vel"],
        old_time,
        new_time
    )

    gripper = interpolate_columns(
        gesture["gripper"],
        old_time,
        new_time
    )

    rotation_interpolator = Slerp(
        old_time,
        Rotation.from_matrix(
            gesture["rotations"]
        )
    )

    rotations = rotation_interpolator(
        new_time
    ).as_matrix()

    return {
        "xyz": xyz,
        "rotations": rotations,
        "trans_vel": trans_vel,
        "rot_vel": rot_vel,
        "gripper": gripper
    }


# -------------------------------------------------------------------------
# CREATE ONE AVERAGED FILE PER GESTURE
# -------------------------------------------------------------------------

fields = [
    "relative_x", "relative_y", "relative_z",
    "relative_qx", "relative_qy", "relative_qz", "relative_qw",
    "relative_trans_vel_x", "relative_trans_vel_y", "relative_trans_vel_z",
    "relative_rot_vel_x", "relative_rot_vel_y", "relative_rot_vel_z",
    "gripper_angle",
]
header = ",".join(["frame"] + [
    f"{arm}_{field}" for arm in ("left", "right") for field in fields
])


def average_arm(examples, number_of_frames):
    resampled = [resample_gesture(example, number_of_frames)
                 for example in examples]
    averages = {
        key: np.mean(np.stack([example[key] for example in resampled]), axis=0)
        for key in ("xyz", "trans_vel", "rot_vel", "gripper")
    }
    quaternions = np.stack([
        Rotation.from_matrix(np.stack([
            example["rotations"][frame] for example in resampled
        ])).mean().as_quat()
        for frame in range(number_of_frames)
    ])
    # Each arm uses its own initial tool coordinate system.
    averages["xyz"] -= averages["xyz"][0]
    return np.column_stack([
        averages["xyz"], quaternions, averages["trans_vel"],
        averages["rot_vel"], averages["gripper"],
    ])


for gesture_name, examples in gesture_examples.items():
    if not examples:
        print(f"{gesture_name}: no examples were found.")
        continue

    # Both tools share the same occurrences and normalized time samples.
    averaged_length = int(round(np.median([
        len(example["left"]["xyz"]) for example in examples
    ])))
    left = average_arm([example["left"] for example in examples], averaged_length)
    right = average_arm([example["right"] for example in examples], averaged_length)
    frames = np.arange(averaged_length).reshape(-1, 1)
    averaged_gesture = np.column_stack([frames, left, right])

    output_file = output_folder / f"{gesture_name}.csv"
    np.savetxt(
        output_file, averaged_gesture, delimiter=",", header=header,
        comments="", fmt=["%d"] + ["%.10f"] * 28,
    )
    print(f"{gesture_name}: saved {averaged_length} frames "
          f"using {len(examples)} paired left/right examples.")

print("\nAveraged left/right gesture files saved to:", output_folder)
