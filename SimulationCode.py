import time
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
import pybullet as p

import math
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation

#global values
#0.6 m is a good location for the skin when the arm is perpendicular to the y axis
# based on plotting the area in which the arm can move
spacing = 0.4
angle = math.asin(spacing/0.6) 
skin_height = 0.01
skin_width_x = 0.15
skin_width_y = 0.10
skinPosX = 0.6 * math.cos(angle) 
skinPosY = spacing - 0.6 * math.sin(angle) 
skinPosZ = 0.744 # raise the skin on a fake table -> this is calculated using code that calculates the lowest amount of error   

#--------------------------------------------------------------

print(f"Spacing: {spacing:.3f} m")
print(f"Angle: {math.degrees(angle):.1f} degrees")
print(
    f"Skin position (XYZ): "
    f"({skinPosX:.4f}, {skinPosY:.4f}, {skinPosZ:.4f}) m"
)
print(
    f"Skin full size (XYZ): "
    f"({2 * skin_width_x:.3f}, "
    f"{2 * skin_width_y:.3f}, "
    f"{2 * skin_height:.3f}) m"
)
print(f"Skin top Z: {skinPosZ + skin_height:.4f} m")

# -----------------------------------------------------------------------------------
runtime_start = time.perf_counter()

#loading in the robotic simulation  ---------------------------------------------------------------
project = Path(
    r"C:\Users\rrebe\Downloads"
    r"\davinci-gym-lite-main\davinci-gym-lite-main"
)
source = project / "urdf" / "davinci_gym_lite.urdf"
mirror_folder = project / "mirrored_robot"
mirror_folder.mkdir(exist_ok=True)



# pathway of the .txts --- loading in the kinemaatics and notation data -----------------------------
base = Path(
    r"C:\Users\rrebe\OneDrive - University of Ottawa\ELG 7113"
)
averaged_folder = (
    base / "averaged_gestures"
)

gestures = {}

for gesture_number in range(1, 12):

    gesture_name = f"G{gesture_number}"
    gesture_file = (
        averaged_folder
        / f"{gesture_name}.csv"
    )

    if not gesture_file.is_file():
        print(
            f"{gesture_name}: averaged file missing."
        )
        continue

    gestures[gesture_name] = np.loadtxt(
        gesture_file,
        delimiter=",",
        skiprows=1,
        ndmin=2
    )
    
#----------------------------------------------------------------------------------------------------------


# Simulate one original arm - downlaoded arm ----------------------------------------

def make_model():
    tree = ET.parse(source)
    root = tree.getroot()

    # Resolve mesh locations.
    for mesh_element in root.iter("mesh"):
        mesh_path = Path(mesh_element.get("filename"))

        if not mesh_path.is_absolute():
            mesh_path = project / mesh_path

        if not mesh_path.is_file():
            raise FileNotFoundError(f"Missing mesh: {mesh_path}")

        mesh_element.set("filename", mesh_path.as_posix())

    # Fill in missing joint-limit attributes.
    for limit in root.iter("limit"):
        if limit.get("effort") is None:
            limit.set("effort", "30")

        if limit.get("velocity") is None:
            limit.set("velocity", "2")
        # Replace collision shapes with shapes covering the visible geometry.
    for link in root.findall("link"):

        # Remove the old simplified collision shapes.
        for old_collision in list(link.findall("collision")):
            link.remove(old_collision)

        if link.get("name") == "arm_4":
            # Box enclosing only the skinny shaft measured from p4_arm4.stl.
            # Local X: 0.040 to 0.3075 m; radius: 0.029 m.
            collision = ET.SubElement(link, "collision")
            ET.SubElement(
                collision, "origin",
                xyz="0.17375 0 -0.1475",
                rpy="0 0 0"
            )
            collision_geometry = ET.SubElement(collision, "geometry")
            ET.SubElement(
                collision_geometry, "box",
                size="0.2675 0.058 0.058"
            )
            continue

        for visual in link.findall("visual"):
            geometry = visual.find("geometry")

            if geometry is None:
                continue

            collision = ET.SubElement(link, "collision")
            visual_origin = visual.find("origin")
            mesh_element = geometry.find("mesh")

            if mesh_element is not None:
                # Mesh paths were already resolved earlier in make_model().
                mesh = trimesh.load(
                    mesh_element.get("filename"),
                    force="mesh"
                )

                scale = np.array(
                    mesh_element.get("scale", "1 1 1").split(),
                    dtype=float
                )

                vertices = np.asarray(mesh.vertices) * scale

                minimum = vertices.min(axis=0)
                maximum = vertices.max(axis=0)

                # A box enclosing the entire visible mesh.
                box_size = np.maximum(maximum - minimum, 1e-6)
                box_center = (minimum + maximum) / 2

                origin_xyz = np.zeros(3)
                origin_rpy = np.zeros(3)

                if visual_origin is not None:
                    origin_xyz = np.array(
                        visual_origin.get("xyz", "0 0 0").split(),
                        dtype=float
                    )
                    origin_rpy = np.array(
                        visual_origin.get("rpy", "0 0 0").split(),
                        dtype=float
                    )

                # Place the box in the same frame as the visible mesh.
                origin_rotation = Rotation.from_euler(
                    "xyz", origin_rpy
                ).as_matrix()

                collision_center = (
                    origin_xyz + origin_rotation @ box_center
                )

                ET.SubElement(
                    collision,
                    "origin",
                    xyz=" ".join(map(str, collision_center)),
                    rpy=" ".join(map(str, origin_rpy))
                )

                collision_geometry = ET.SubElement(
                    collision, "geometry"
                )

                ET.SubElement(
                    collision_geometry,
                    "box",
                    size=" ".join(map(str, box_size))
                )

            else:
                # Copy visible cylinders, spheres and boxes directly.
                if visual_origin is not None:
                    collision.append(
                        ET.fromstring(ET.tostring(visual_origin))
                    )

                collision.append(
                    ET.fromstring(ET.tostring(geometry))
                )

    output = project / "original_resolved.urdf"
    tree.write(output, encoding="utf-8", xml_declaration=True)

    return output


original_path = make_model()

p.connect(p.GUI)
p.setGravity(0, 0, 0)


#ROBOTIC SIMULATION ----------------------------------------------------------------------


mirrored_path = project / "mirrored_robot" / "mirrored.urdf"

if not mirrored_path.is_file():
    raise FileNotFoundError(mirrored_path)

# Original arm at positive Y.
robot_id = p.loadURDF(
    str(original_path),
    basePosition=[0, spacing, 0],
    baseOrientation=p.getQuaternionFromEuler(
        [0, 0, math.pi + math.pi/2 - angle]
    ),
    useFixedBase=True
)

# Mirrored arm at negative Y.
#robot_id_2 = p.loadURDF(
 #   str(mirrored_path),
  #  basePosition=[0, -spacing, 0],
   # baseOrientation=p.getQuaternionFromEuler(
        #[0, 0, -math.pi/2 + angle]
   # ),
  #  useFixedBase=True
#)

p.resetDebugVisualizerCamera(
    cameraDistance=3,
    cameraYaw=90,
    cameraPitch=-25,
    cameraTargetPosition=[0, 0, 0.7]
)

#----------------------------------------------------------------------------------------------------------

#adding a piece of fake skin ------------------------------------------------------------------------
skin_size = [skin_width_x, skin_width_y, skin_height]  # Half-sizes: full block is 30 × 20 × 2 cm
#center location of the fake skin 
skin_position = [skinPosX, skinPosY, skinPosZ]   #somewhere in the overlapped reachable space  # Centre position in metres; adjust as needed

skin_visual = p.createVisualShape(
    p.GEOM_BOX,
    halfExtents=skin_size
)

skin_collision = p.createCollisionShape(
    p.GEOM_BOX,
    halfExtents=skin_size
)

skin = p.createMultiBody(
    baseMass=0,  # Fixed in place
    baseCollisionShapeIndex=skin_collision,
    baseVisualShapeIndex=skin_visual,
    basePosition=skin_position
)

#--------------------------------------------------------------------------------------------------



# -------------------------------------------------------------------------
# BUILD ONE CONTINUOUS TRAJECTORY FROM MULTIPLE GESTURES
# -------------------------------------------------------------------------

robot = robot_id

# Find the robot's tool-tip link before transforming the gestures.
tool_tip = None

for joint in range(p.getNumJoints(robot)):
    link_name = p.getJointInfo(robot, joint)[12].decode()

    if link_name == "tool_tip":
        tool_tip = joint
        break

if tool_tip is None:
    raise ValueError("Could not find the tool_tip link.")


# -------------------------------------------------------------------------
# Starting position — used only for the first gesture
# -------------------------------------------------------------------------

first_start_position = np.array([
    skinPosX + skin_width_x - 0.1,
    skinPosY + skin_width_y / 2,
    skinPosZ + skin_height
])


# Get the robot tool's current starting orientation.
state = p.getLinkState(
    robot,
    tool_tip,
    computeForwardKinematics=True
)

first_start_rotation = Rotation.from_quat(
    state[5]
).as_matrix()


# -------------------------------------------------------------------------
# Specify the order in which gestures will be played
# -------------------------------------------------------------------------

gesture_order = ["G1", "G5", "G8", "G2", "G3", "G6", "G4",
                 "G10", "G2", "G3", "G6", "G9", "G4", "G2",
                 "G3", "G6", "G4", "G2", "G3", "G6", "G11"]

# Starting pose of the first gesture.
current_position = first_start_position.copy()
current_rotation = first_start_rotation.copy()

continuous_trajectory = []
trajectory_gestures = []
frame_counter = 0


# -------------------------------------------------------------------------
# Transform and connect every gesture
# -------------------------------------------------------------------------

for gesture_name in gesture_order:

    if gesture_name not in gestures:
        raise ValueError(
            f"No averaged data were found for {gesture_name}. "
            f"Available gestures: {sorted(gestures)}"
    )

    #reuse the complete avg traj for this gesture 
    gesture = gestures[gesture_name].copy()

    print(
        f"Adding {gesture_name} using its saved trajectory"
    )

    # -------------------------------------------------------------
    # Recorded starting pose of this gesture
    # -------------------------------------------------------------

    recorded_start_position = gesture[0, 1:4]

    recorded_start_rotation = Rotation.from_quat(
        gesture[0, 4:8]
    ).as_matrix()


    # Rotate the recorded coordinate system so that the gesture's
    # first orientation matches the preceding gesture's final
    # orientation.
    A = current_rotation @ recorded_start_rotation.T


    # -------------------------------------------------------------
    # Transform position
    # -------------------------------------------------------------

    recorded_positions = gesture[:, 1:4]

    # Express every point relative to this gesture's first point.
    relative_positions = (
        recorded_positions - recorded_start_position
    )

    # Place the gesture at the endpoint of the preceding gesture.
    global_positions = (
        relative_positions @ A.T
        + current_position
    )

    gesture[:, 1:4] = global_positions


    # -------------------------------------------------------------
    # Transform orientation
    # -------------------------------------------------------------

    recorded_rotations = Rotation.from_quat(
        gesture[:, 4:8]
    ).as_matrix()

    global_rotations = A @ recorded_rotations

    gesture[:, 4:8] = Rotation.from_matrix(
        global_rotations
    ).as_quat()


    # -------------------------------------------------------------
    # Transform translational velocity
    # -------------------------------------------------------------

    gesture[:, 8:11] = gesture[:, 8:11] @ A.T


    # Give the combined trajectory continuous frame numbers.
    number_of_frames = len(gesture)

    gesture[:, 0] = np.arange(
        frame_counter,
        frame_counter + number_of_frames
    )

    frame_counter += number_of_frames


    # Add this occurrence to the overall trajectory.
    continuous_trajectory.append(gesture)
    trajectory_gestures.extend([gesture_name] * len(gesture))


    # The final pose becomes the next gesture's starting pose.
    current_position = gesture[-1, 1:4].copy()

    current_rotation = Rotation.from_quat(
        gesture[-1, 4:8]
    ).as_matrix()


    print(
        f"  Start position: {gesture[0, 1:4]}"
    )
    print(
        f"  End position:   {gesture[-1, 1:4]}"
    )


# Combine all occurrences into one trajectory.
trajectory = np.vstack(continuous_trajectory)

print("\nContinuous trajectory created.")
print("Gesture order:", gesture_order)
print("Total trajectory points:", len(trajectory))
print("Initial position:", trajectory[0, 1:4])
print("Final position:", trajectory[-1, 1:4])

print("Mapped start:", trajectory[0, 1:4])
print("Simulated start:", first_start_position)
print(
    "Starting position mismatch (mm):",
    np.linalg.norm(trajectory[0, 1:4] - first_start_position) * 1000
)


# inverse kinematics to move the tool tip ------------------------------------------------------------------

# -------------------------------------------------------------------------
# INVERSE KINEMATICS
# -------------------------------------------------------------------------

# The variables below were already created in the continuous-trajectory
# section:
#
# robot = robot_id
# tool_tip = tool-tip link index
# trajectory = combined G1, G5, G8 and G2 trajectory

joints = []
lower = []
upper = []

# Find every movable robot joint and save its movement limits.
for joint in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, joint)

    joint_type = info[2]

    if joint_type in (
        p.JOINT_REVOLUTE,
        p.JOINT_PRISMATIC
    ):
        lo = info[8]
        hi = info[9]

        # Some continuous revolute joints do not have valid limits
        # in the URDF, so use -pi to +pi.
        if lo > hi:
            if joint_type == p.JOINT_REVOLUTE:
                lo = -np.pi
                hi = np.pi
            else:
                raise ValueError(
                    f"Invalid limits for joint {joint}"
                )

        joints.append(joint)
        lower.append(lo)
        upper.append(hi)

# Convert the Python lists into NumPy arrays.
lower = np.array(lower)
upper = np.array(upper)

# Assumes the recorded tool axes match the URDF tool-tip axes.
tool_alignment = Rotation.identity()

# Each playback result is stored here.
left_results = []

sample_rate = 30.0
playback_speed = 1.0
playback_start = time.monotonic()
first_frame = trajectory[0, 0]

# ----------------------------------------------------------
# Read each row and reproduce the left-tool pose.
# ----------------------------------------------------------
# ----------------------------------------------------------
# SKIN-SURFACE SAFETY BOUNDARY
# ----------------------------------------------------------

# Top surface and horizontal boundaries of the skin.
skin_top_z = skinPosZ + skin_height

skin_min_x = skinPosX - skin_width_x
skin_max_x = skinPosX + skin_width_x

skin_min_y = skinPosY - skin_width_y
skin_max_y = skinPosY + skin_width_y

# Keep protected robot links slightly above the skin.
skin_clearance = 0.002  # 2 mm

# Identify links belonging to the surgical instrument.
# Only this link is allowed below the skin surface.
# Add other link indices only after verifying they are tool parts.
allowed_tool_names = {"instrument_house", "instrument_neck", "tool_tip"}
tool_links = {
    index for index in range(p.getNumJoints(robot))
    if p.getJointInfo(robot, index)[12].decode() in allowed_tool_names
}

print("\nSkin boundary link rules:")

for link_index in range(p.getNumJoints(robot)):
    name = p.getJointInfo(robot, link_index)[12].decode()
    rule = "ALLOWED into skin" if link_index in tool_links else "PROTECTED"
    print(f"  {link_index}: {name} — {rule}")

    # Do not silently accept protected links with no collision geometry.
    if link_index not in tool_links:
        collision_shapes = p.getCollisionShapeData(robot, link_index)

        if not collision_shapes:
            has_visual_shape = any(
                shape[1] == link_index
                for shape in p.getVisualShapeData(robot)
            )

            if has_visual_shape:
                raise RuntimeError(
                    f"Link {link_index} ({name}) has a visible shape "
                    "but no collision geometry. "
                    "Its URDF needs collision geometry."
                )

            print(f"  Skipping empty link: {link_index} ({name})")

print("Links allowed to enter the skin:")

for link_index in sorted(tool_links):
    link_name = p.getJointInfo(
        robot,
        link_index
    )[12].decode()

    print(f"  {link_index}: {link_name}")


# Use a separate physics connection: test poses never appear in the GUI.
checker_client = p.connect(p.DIRECT)
checker_base_position, checker_base_orientation = p.getBasePositionAndOrientation(robot)
checker_robot = p.loadURDF(
    str(original_path),
    basePosition=checker_base_position,
    baseOrientation=checker_base_orientation,
    useFixedBase=True,
    physicsClientId=checker_client
)

if p.getNumJoints(checker_robot, physicsClientId=checker_client) != p.getNumJoints(robot):
    raise RuntimeError("Checker robot does not match the displayed robot.")

checker_links = [
    index for index in range(-1, p.getNumJoints(robot))
    if index not in tool_links
    and p.getCollisionShapeData(
        checker_robot, index, physicsClientId=checker_client
    )
]


def arm_is_above_skin(candidate_joint_values):
    """Check every protected link using the invisible robot only."""
    candidate_joint_values = np.asarray(candidate_joint_values, dtype=float)
    if candidate_joint_values.shape != (len(joints),):
        raise ValueError("Unexpected number of candidate joint values.")
    if not np.all(np.isfinite(candidate_joint_values)):
        raise ValueError("Candidate joint values contain NaN or infinity.")

    for joint, value in zip(joints, candidate_joint_values):
        p.resetJointState(
            checker_robot, joint, float(value),
            physicsClientId=checker_client
        )

    p.performCollisionDetection(physicsClientId=checker_client)
    unsafe_links = []

    for link_index in checker_links:
        aabb_min, aabb_max = p.getAABB(
            checker_robot, link_index, physicsClientId=checker_client
        )
        overlaps_x = (
            aabb_max[0] >= skin_min_x and aabb_min[0] <= skin_max_x
        )
        overlaps_y = (
            aabb_max[1] >= skin_min_y and aabb_min[1] <= skin_max_y
        )
        if overlaps_x and overlaps_y and aabb_min[2] < skin_top_z + skin_clearance:
            link_name = (
                "tower_link" if link_index == -1 else
                p.getJointInfo(
                    checker_robot, link_index,
                    physicsClientId=checker_client
                )[12].decode()
            )
            unsafe_links.append({
                "index": link_index,
                "name": link_name,
                "bottom_z": aabb_min[2]
            })

    # This return is OUTSIDE the loop: all protected links are checked
    return len(unsafe_links) == 0, unsafe_links



# Maximum number of DIFFERENT starting configurations per trajectory frame
max_ik_attempts = 70 # not  unlimited to speed up the processing time of the code  
max_position_error_mm = 1.0
max_orientation_error_deg = 1.0
ik_rng = np.random.default_rng(42)
rejected_frames = 0
total_ik_attempts = 0


def find_safe_ik(target_xyz, target_rotation, preferred_joints):
    """Try different starting configurations on the invisible robot."""
    failure_counts = {"boundary": 0, "pose_error": 0, "invalid": 0}
    blocking_links = set()
    best_position_error = float("inf")
    closest_boundary_mm = float("inf")

    for attempt in range(1, max_ik_attempts + 1):
        if attempt == 1:
            seed = preferred_joints.copy()
        elif attempt == 2:
            seed = (lower + upper) / 2
        elif attempt <= max_ik_attempts // 2:
            # Explore around the last displayed pose first
            seed = np.clip(
                preferred_joints + ik_rng.normal(
                    0.0, 0.15, len(joints)
                ) * (upper - lower), lower, upper
            )
        else:
            # Explore other arm configurations across the joint limits.
            seed = ik_rng.uniform(lower, upper)

        for joint, value in zip(joints, seed):
            p.resetJointState(
                checker_robot, joint, float(value),
                physicsClientId=checker_client
            )

        solution = np.asarray(p.calculateInverseKinematics(
            bodyUniqueId=checker_robot,
            endEffectorLinkIndex=tool_tip,
            targetPosition=target_xyz.tolist(),
            targetOrientation=target_rotation.as_quat().tolist(),
            lowerLimits=lower.tolist(),
            upperLimits=upper.tolist(),
            jointRanges=(upper - lower).tolist(),
            restPoses=seed.tolist(),
            maxNumIterations=500,
            residualThreshold=1e-6,
            physicsClientId=checker_client
        ), dtype=float)

        if solution.shape != lower.shape or not np.all(np.isfinite(solution)):
            failure_counts["invalid"] += 1
            continue

        candidate = np.clip(solution, lower, upper)
        safe, unsafe_links = arm_is_above_skin(candidate)
        # arm_is_above_skin leaves the invisible robot at this candidate.
        state = p.getLinkState(
            checker_robot, tool_tip, computeForwardKinematics=True,
            physicsClientId=checker_client
        )
        position_error = np.linalg.norm(np.asarray(state[4]) - target_xyz) * 1000
        orientation_error = np.degrees((
            target_rotation.inv() * Rotation.from_quat(state[5])
        ).magnitude())
        if not np.isfinite(position_error) or not np.isfinite(orientation_error):
            failure_counts["invalid"] += 1
            continue
        best_position_error = min(best_position_error, position_error)

        if not safe:
            failure_counts["boundary"] += 1
            blocking_links.update(link["name"] for link in unsafe_links)
            worst_violation = max(
                (skin_top_z + skin_clearance - link["bottom_z"]) * 1000
                for link in unsafe_links
            )
            closest_boundary_mm = min(closest_boundary_mm, worst_violation)

        pose_ok = (
            position_error <= max_position_error_mm
            and orientation_error <= max_orientation_error_deg
        )
        if not pose_ok:
            failure_counts["pose_error"] += 1

        if safe and pose_ok:
            return candidate, bool(np.any(np.abs(solution - candidate) > 1e-8)), attempt, ""

    message = (
        f"boundary failures={failure_counts['boundary']}, "
        f"pose-error failures={failure_counts['pose_error']}, "
        f"invalid results={failure_counts['invalid']}; "
        f"best position error={best_position_error:.3f} mm"
    )
    if blocking_links:
        message += (
            f"; blocking links={', '.join(sorted(blocking_links))}; "
            f"smallest worst bounding-box violation={closest_boundary_mm:.3f} mm"
        )
    return None, False, max_ik_attempts, message


for row in trajectory:

    if not p.isConnected():
        break

    # Read this trajectory row
    # These positions have already been transformed into world coordinates. 
    frame = int(row[0])
    target_xyz = row[1:4] 
    recorded_quaternion = row[4:8]
    recorded_trans_vel = row[8:11]
    recorded_rot_vel = row[11:14]
    recorded_gripper_angle = float(row[14])

    # Calculate the desired tool orientation.
    target_rotation = (
        Rotation.from_quat(recorded_quaternion)
        * tool_alignment
    )

    target_quaternion = target_rotation.as_quat()


    # ------------------------------------------------------
    # Wait until this frame's playback time
    # ------------------------------------------------------

    scheduled_time = playback_start + (
        (frame - first_frame)
        / (sample_rate * playback_speed)
    )

    remaining = scheduled_time - time.monotonic()

    if remaining > 0:
        time.sleep(remaining)


    # ------------------------------------------------------
    # Search for an acceptable IK solution
    # ------------------------------------------------------

    # Prefer the current displayed configuration on the first attempt.
    current_joints = np.array([
        p.getJointState(robot, joint)[0]
        for joint in joints
    ])

    # PyBullet performs up to 500 internal iterations
    # The robot does not move during this calculation
    
    joint_targets, limits_clipped, attempts, failure_message = find_safe_ik(
        target_xyz, target_rotation, current_joints
    )
    total_ik_attempts += attempts

    if joint_targets is None:
        rejected_frames += 1
        print(f"Frame {frame} rejected after {attempts} IK attempts: {failure_message}")
        # Leave the displayed robot at its previous configuration.
        continue

    if attempts > 1:
        print(f"Frame {frame}: acceptable IK pose found on attempt {attempts}")

    # ------------------------------------------------------
    # The candidate is safe, so move all joints
    # ------------------------------------------------------

    for joint, value in zip(
        joints,
        joint_targets
    ):
        p.resetJointState(
            robot,
            joint,
            float(value)
        )

    # This must be outside the joint loop.
    # All joints have now been moved.
    state = p.getLinkState(
        robot,
        tool_tip,
        computeForwardKinematics=True
    )

    actual_xyz = np.array(state[4])
    actual_quaternion = np.array(state[5])

    # ------------------------------------------------------
    # Calculate error once for this trajectory frame
    # ------------------------------------------------------

    position_error_mm = (
        np.linalg.norm(
            actual_xyz - target_xyz
        ) * 1000
    )

    orientation_error_deg = np.degrees(
        (
            target_rotation.inv()
            * Rotation.from_quat(
                actual_quaternion
            )
        ).magnitude()
    )

    print(
        f"Frame {frame}: "
        f"position error {position_error_mm:.3f} mm, "
        f"orientation error {orientation_error_deg:.3f} degrees"
    )

    # ------------------------------------------------------
    # Report joints near their limits
    # ------------------------------------------------------

    for joint, value, lo, hi in zip(
        joints,
        joint_targets,
        lower,
        upper
    ):
        if min(
            abs(value - lo),
            abs(value - hi)
        ) < 0.001:

            name = p.getJointInfo(
                robot,
                joint
            )[1].decode()

            print(
                f"  Near limit: {name}, "
                f"value={value:.4f}, "
                f"limits=[{lo:.4f}, {hi:.4f}]"
            )

    # ------------------------------------------------------
    # Store one result for this trajectory frame
    # ------------------------------------------------------

    left_results.append({
        "frame": frame,
        "target_xyz": target_xyz.copy(),
        "actual_xyz": actual_xyz.copy(),
        "target_quaternion": target_quaternion.copy(),
        "actual_quaternion": actual_quaternion.copy(),
        "recorded_trans_vel": recorded_trans_vel.copy(),
        "recorded_rot_vel": recorded_rot_vel.copy(),
        "recorded_gripper_angle": recorded_gripper_angle,
        "joint_targets": joint_targets.copy(),
        "limits_clipped": limits_clipped,
        "position_error_mm": position_error_mm,
        "orientation_error_deg": orientation_error_deg
    })
print("\nTotal IK attempts:", total_ik_attempts)
print("Frames accepted:", len(left_results))
print("Frames rejected:", rejected_frames)
if not left_results:
    print("No acceptable pose found. The displayed robot stayed at its initial pose.")

# ----------------------------------------------------------
# Print accuracy.
# ----------------------------------------------------------
if left_results:
    position_errors = [
        result["position_error_mm"] for result in left_results
    ]
    orientation_errors = [
        result["orientation_error_deg"] for result in left_results
    ]

    max_error_index = np.argmax(position_errors)
    max_error_frame = left_results[max_error_index]["frame"]

    trajectory_index = np.flatnonzero(
        trajectory[:, 0] == max_error_frame
    )[0]

    max_error_gesture = trajectory_gestures[trajectory_index]
    
    print("\nG1 left-tool playback complete")
    print("Frames played:", len(left_results))
    print(
        f"Mean position error: {np.mean(position_errors):.3f} mm"
    )
    print(
        f"Maximum position error: {np.max(position_errors):.3f} mm"
    )
    print(
        f"Location of Maximum Position error: "
        f"gesture {max_error_gesture}"
    )
    print(
        "Mean orientation error: "
        f"{np.mean(orientation_errors):.3f} degrees"
    )
    print(
        "Frames with clipped joint targets:",
        sum(result["limits_clipped"] for result in left_results)
    )

# --------------------------------------------------------------------------------------------------------
runtime_seconds = time.perf_counter() - runtime_start

print(
    f"{runtime_seconds / 60:.2f} minutes"
)

# keep open FINAL - ----------------------------- --------------------------- -----------------------
# Keep the simulation window open.
try:
    while p.isConnected():
        time.sleep(1 / 60)

except KeyboardInterrupt:
    pass

finally:
    if p.isConnected(checker_client):
        p.disconnect(checker_client)
    if p.isConnected():
        p.disconnect()
