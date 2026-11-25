import time
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
import pybullet as p
import math
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.spatial.transform import Rotation

#global values
#0.6 m is a good location for the skin when the arm is perpendicular to the y axis
# based on plotting the area in which the arm can move
robotBaseHeight = 1 # [m]  # [m] 
arm_rot = 0.1 # 0.29 [rad] -> this is the rotation along the y-axis of each arm 
spacing = 0.3 + (robotBaseHeight * math.tan(arm_rot)) # [m] 
angle = math.asin(spacing/0.6) # [rad] -> this is the rotation along the z-axis 

# half the actual width length and height of the skin 
skin_height = 0.01 # [m]
skin_width_x = 0.25 # [m]
skin_width_y = 0.30 # [m] 
skinPosX = 0.6 * math.cos(angle) # [m] 
skinPosY = spacing - 0.6 * math.sin(angle) # [m] 
skinPosZ = 0.744 # raise the skin on a fake table -> this is calculated using code that calculates the lowest amount of error   


#manually excel calculated values - fix this so it calcs every turn???
#written in the kinematics coord system 
x_diff = 61.299/1000 # [m] 
y_diff = 19.881/1000 # [m]
z_diff = 21.708/1000 # [m]

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

def make_model(input_source=source, output_name="original_resolved.urdf", shaft_only=True):
    tree = ET.parse(input_source)
    root = tree.getroot()

    # Resolve mesh locations.
    for mesh_element in root.iter("mesh"):
        mesh_path = Path(mesh_element.get("filename"))

        if not mesh_path.is_absolute():
            candidates = [project / mesh_path, input_source.parent / mesh_path]
            mesh_path = next((path for path in candidates if path.is_file()), candidates[0])

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

        if shaft_only and link.get("name") == "arm_4":
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

    output = project / output_name
    tree.write(output, encoding="utf-8", xml_declaration=True)

    return output


original_path = make_model()

p.connect(p.GUI)
p.setGravity(0, 0, 0)


#ROBOTIC SIMULATION ----------------------------------------------------------------------


mirrored_path = project / "mirrored_robot" / "mirrored.urdf"

if not mirrored_path.is_file():
    raise FileNotFoundError(mirrored_path)

# Prepare collision boxes for the mirrored model too.
# Its mirrored meshes use their own coordinates, so do not reuse the
# original arm_4 shaft coordinates on this different model.
mirrored_path = make_model(mirrored_path, "mirrored_resolved.urdf", shaft_only=False)

# Original arm at positive Y.
robot_id = p.loadURDF(
    str(original_path),
    basePosition=[0, -spacing, skinPosZ + robotBaseHeight*2], #make the 'top' of the support one full space above the skin
    baseOrientation=p.getQuaternionFromEuler(
        [-arm_rot, math.pi,  math.pi +math.pi/2 + angle]
    ),
    useFixedBase=True
)

# Mirrored arm at negative Y.
robot_id_2 = p.loadURDF(
    str(mirrored_path),
    basePosition=[0, spacing, skinPosZ + robotBaseHeight*2],
    baseOrientation=p.getQuaternionFromEuler(
        [-arm_rot, math.pi, -math.pi +math.pi/2 - angle]
    ),
    useFixedBase=True
)

# Colour the entire second robot green, including its base.
for link in range(-1, p.getNumJoints(robot_id_2)):
    p.changeVisualShape(
        robot_id_2,
        link,
        rgbaColor=[0.0, 0.8, 0.0, 1.0]
    )

# Slightly bend the second arm away from its default pose.
# Add 10 degrees to every rotational joint of the second robot.
for joint in range(p.getNumJoints(robot_id_2)):
    info = p.getJointInfo(robot_id_2, joint)

    if info[2] == p.JOINT_REVOLUTE:
        current_angle = p.getJointState(robot_id_2, joint)[0]
        new_angle = current_angle + math.radians(25)

        if info[8] <= info[9]:
            new_angle = float(np.clip(new_angle, info[8], info[9]))

        p.resetJointState(robot_id_2, joint, new_angle)


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



# Both tools share one gesture order and frame timeline.
gesture_order = ["G1", "G5", "G8", "G2", "G3", "G6", "G4",
                 "G10", "G2", "G3", "G6", "G9", "G4", "G2",
                 "G3", "G6", "G4", "G2", "G3", "G6", "G11"]
max_ik_attempts = 100
max_position_error_mm = 1.0
max_orientation_error_deg = 1.0
sample_rate = 30.0
playback_speed = 1.0
skin_top_z = skinPosZ + skin_height
skin_min_x, skin_max_x = skinPosX - skin_width_x, skinPosX + skin_width_x
skin_min_y, skin_max_y = skinPosY - skin_width_y, skinPosY + skin_width_y
skin_clearance = 0.002
allowed_tool_names = {"instrument_house", "instrument_neck", "tool_tip"}
checker_client = p.connect(p.DIRECT)


class Arm:
    def __init__(self, name, robot, model_path, random_seed):
        self.name = name
        self.robot = robot
        self.joints = []
        lower, upper = [], []
        self.tool_tip = None
        self.ik_rng = np.random.default_rng(random_seed)
        self.results = []
        self.total_attempts = 0
        self.ik_failures = 0
        self.held_frames = 0
        visual_links = {shape[1] for shape in p.getVisualShapeData(robot)}
        for index in range(p.getNumJoints(robot)):
            info = p.getJointInfo(robot, index)
            if info[12].decode() == "tool_tip":
                self.tool_tip = index
            if info[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                lo, hi = info[8], info[9]
                if lo > hi:
                    if info[2] != p.JOINT_REVOLUTE:
                        raise ValueError(f"{name}: invalid joint limits")
                    lo, hi = -np.pi, np.pi
                self.joints.append(index)
                lower.append(lo)
                upper.append(hi)
        if self.tool_tip is None:
            raise ValueError(f"{name}: no tool_tip link")
        self.lower, self.upper = np.array(lower), np.array(upper)
        position, orientation = p.getBasePositionAndOrientation(robot)
        self.checker_robot = p.loadURDF(
            str(model_path), basePosition=position, baseOrientation=orientation,
            useFixedBase=True, physicsClientId=checker_client)
        self.checker_links = []
        for index in range(-1, p.getNumJoints(robot)):
            name_link = (p.getBodyInfo(robot)[0].decode() if index == -1
                         else p.getJointInfo(robot, index)[12].decode())
            if name_link in allowed_tool_names:
                continue
            if p.getCollisionShapeData(self.checker_robot, index,
                                       physicsClientId=checker_client):
                self.checker_links.append(index)
            elif index in visual_links:
                raise RuntimeError(f"{name}: {name_link} has no collision geometry")

    def arm_is_above_skin(self, candidate_joint_values):
        """Check every protected link using the invisible robot only."""
        candidate_joint_values = np.asarray(candidate_joint_values, dtype=float)
        if candidate_joint_values.shape != (len(self.joints),):
            raise ValueError("Unexpected number of candidate joint values.")
        if not np.all(np.isfinite(candidate_joint_values)):
            raise ValueError("Candidate joint values contain NaN or infinity.")

        for joint, value in zip(self.joints, candidate_joint_values):
            p.resetJointState(
                self.checker_robot, joint, float(value),
                physicsClientId=checker_client
            )

        p.performCollisionDetection(physicsClientId=checker_client)
        unsafe_links = []

        for link_index in self.checker_links:
            aabb_min, aabb_max = p.getAABB(
                self.checker_robot, link_index, physicsClientId=checker_client
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
                        self.checker_robot, link_index,
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

    def find_safe_ik(self, target_xyz, target_rotation, preferred_joints):
        """Try different starting configurations on the invisible robot."""
        failure_counts = {"boundary": 0, "pose_error": 0, "invalid": 0}
        blocking_links = set()
        best_position_error = float("inf")
        closest_boundary_mm = float("inf")

        for attempt in range(1, max_ik_attempts + 1):
            if attempt == 1:
                seed = preferred_joints.copy()
            elif attempt == 2:
                seed = (self.lower + self.upper) / 2
            elif attempt <= max_ik_attempts // 2:
                # Explore around the last displayed pose first
                seed = np.clip(
                    preferred_joints + self.ik_rng.normal(
                        0.0, 0.15, len(self.joints)
                    ) * (self.upper - self.lower), self.lower, self.upper
                )
            else:
                # Explore other arm configurations across the joint limits.
                seed = self.ik_rng.uniform(self.lower, self.upper)

            for joint, value in zip(self.joints, seed):
                p.resetJointState(
                    self.checker_robot, joint, float(value),
                    physicsClientId=checker_client
                )

            solution = np.asarray(p.calculateInverseKinematics(
                bodyUniqueId=self.checker_robot,
                endEffectorLinkIndex=self.tool_tip,
                targetPosition=target_xyz.tolist(),
                targetOrientation=target_rotation.as_quat().tolist(),
                lowerLimits=self.lower.tolist(),
                upperLimits=self.upper.tolist(),
                jointRanges=(self.upper - self.lower).tolist(),
                restPoses=seed.tolist(),
                maxNumIterations=100,
                residualThreshold=1e-6,
                physicsClientId=checker_client
            ), dtype=float)

            if solution.shape != self.lower.shape or not np.all(np.isfinite(solution)):
                failure_counts["invalid"] += 1
                continue

            candidate = np.clip(solution, self.lower, self.upper)
            safe, unsafe_links = self.arm_is_above_skin(candidate)
            # arm_is_above_skin leaves the invisible robot at this candidate.
            state = p.getLinkState(
                self.checker_robot, self.tool_tip, computeForwardKinematics=True,
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

def build_trajectory(arm, column_start, first_position):
    state = p.getLinkState(arm.robot, arm.tool_tip, computeForwardKinematics=True)
    current_rotation = Rotation.from_quat(state[5]).as_matrix()
    current_position = np.array(first_position, dtype=float)
    parts, labels = [], []
    frame_counter = 0
    for gesture_name in gesture_order:
        if gesture_name not in gestures:
            raise ValueError(f"Missing gesture {gesture_name}")
        data = gestures[gesture_name]
        if data.shape[1] != 29:
            raise ValueError(
                f"{gesture_name}: expected 29 columns (frame + left + right). "
                "Run AverageGesturesBothArms.py first.")
        # Local trajectory layout: frame + this tool's 14 values.
        gesture = np.column_stack([data[:, 0], data[:, column_start:column_start+14]])
        # Recorded axes → simulation ground axes:
        # X_ground = -X_kin
        # Y_ground =  Y_kin
        # Z_ground = -Z_kin
        A = np.array([
            [-1.0, 0.0,  0.0],
            [ 0.0, 1.0,  0.0],
            [ 0.0, 0.0, -1.0]
        ])
        gesture[:, 1:4] = (gesture[:, 1:4] - gesture[0, 1:4]) @ A.T + current_position
        gesture[:, 4:8] = Rotation.from_matrix(
            A @ Rotation.from_quat(gesture[:, 4:8]).as_matrix()).as_quat()
        gesture[:, 8:11] = gesture[:, 8:11] @ A.T
        gesture[:, 11:14] = gesture[:, 11:14] @ A.T
        gesture[:, 0] = np.arange(frame_counter, frame_counter + len(gesture))
        frame_counter += len(gesture)
        parts.append(gesture)
        labels.extend([gesture_name] * len(gesture))
        current_position = gesture[-1, 1:4].copy()
        current_rotation = Rotation.from_quat(gesture[-1, 4:8]).as_matrix()
    return np.vstack(parts), labels


def print_summary(arm, labels):
    print(f"\n{arm.name} arm:")
    print("Total IK attempts:", arm.total_attempts)
    print("Frames moved:", len(arm.results))
    print("Frames with no acceptable IK:", arm.ik_failures)
    print("Frames held because the other arm failed:", arm.held_frames)
    if arm.results:
        errors = np.array([r["position_error_mm"] for r in arm.results])
        worst = arm.results[int(np.argmax(errors))]
        print(f"Mean position error: {errors.mean():.3f} mm")
        print(f"Maximum position error: {errors.max():.3f} mm")
        print("Location of maximum position error: gesture", labels[worst["frame"]])
        print("Mean orientation error:",
              f"{np.mean([r['orientation_error_deg'] for r in arm.results]):.3f} degrees")
        print("Frames with clipped joint targets:",
              sum(r["limits_clipped"] for r in arm.results))


def print_gesture_results(arm, labels):
    """
    Print one row per gesture for this arm.

    Acceptance (%) = executed frames / requested frames × 100.

    Position and orientation errors are calculated ONLY for
    successfully executed frames.

    Rejected frames and frames held because the other arm failed
    are included in the requested count, but not the executed count.
    """

    # Keep gestures in the order they first appear in the sequence.
    unique_gestures = list(dict.fromkeys(labels))

    print(f"\n{arm.name.upper()} ARM — RESULTS BY GESTURE")

    # Set column widths so the printed table lines up.
    header = (
        f"{'Gesture':<9}"
        f"{'Requested':>11}"
        f"{'Executed':>11}"
        f"{'Accepted (%)':>15}"
        f"{'Mean pos (mm)':>16}"
        f"{'Mean ori (deg)':>17}"
    )

    print(header)
    print("-" * len(header))

    for gesture_name in unique_gestures:

        # Count all requested frames belonging to this gesture,
        # including repeated occurrences of the same gesture.
        requested = labels.count(gesture_name)

        # Each stored result contains its original frame number.
        # Use that number to find the corresponding gesture label.
        gesture_results = [
            result
            for result in arm.results
            if labels[result["frame"]] == gesture_name
        ]

        executed = len(gesture_results)
        acceptance = 100.0 * executed / requested

        if executed > 0:
            mean_position = np.mean([
                result["position_error_mm"]
                for result in gesture_results
            ])

            mean_orientation = np.mean([
                result["orientation_error_deg"]
                for result in gesture_results
            ])

            position_text = f"{mean_position:.3f}"
            orientation_text = f"{mean_orientation:.3f}"

        else:
            # No executed frames means there is no measured error.
            # Display N/A instead of misleadingly reporting zero.
            position_text = "N/A"
            orientation_text = "N/A"

        print(
            f"{gesture_name:<9}"
            f"{requested:>11}"
            f"{executed:>11}"
            f"{acceptance:>15.1f}"
            f"{position_text:>16}"
            f"{orientation_text:>17}"
        )


try:
    left = Arm("Left", robot_id, original_path, 42)
    right = Arm("Right", robot_id_2, mirrored_path, 43)
    # Independent starting points 10 cm apart on the skin.
    # These are chosen placements; the averaged files do not retain the
    # recorded separation between the two tools.
    # left tool, arbitrary start position on the skin
    left_start_x = skinPosX + x_diff/2 #skin_width_x - 0.1
    left_start_y = skinPosY + y_diff/2 #skin_width_y / 2
    left_start_z = skin_top_z + z_diff

    right_start_x = left_start_x - x_diff
    right_start_y = left_start_y - y_diff
    right_start_z = left_start_z 
    
    left_start = [left_start_x,
                  left_start_y,
                  left_start_z]
    # right position moved based on the initial start position from each downloaded kinematics
    right_start = [right_start_x,
                   right_start_y,
                   right_start_z]
    
    # Recorded RIGHT motion goes to the LEFT robot
    left_trajectory, labels = build_trajectory(left, 1, left_start)
    # Recorded LEFT motion goes to the RIGHT robot
    right_trajectory, right_labels = build_trajectory(right, 15, right_start)
    
    print("Left starting XYZ:", left_start)
    print("Right starting XYZ:", right_start)
    print("Total paired frames:", len(left_trajectory))
    paired_rejections = 0
    playback_start = time.monotonic()
    for left_row, right_row in zip(left_trajectory, right_trajectory):
        if not p.isConnected(0):
            break
        frame = int(left_row[0])
        remaining = playback_start + frame / (sample_rate * playback_speed) - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        plans = []
        for arm, row in ((left, left_row), (right, right_row)):
            target_xyz = row[1:4]
            target_rotation = Rotation.from_quat(row[4:8])
            preferred = np.array([p.getJointState(arm.robot, j)[0] for j in arm.joints])
            candidate, clipped, attempts, message = arm.find_safe_ik(
                target_xyz, target_rotation, preferred)
            arm.total_attempts += attempts
            if candidate is None:
                arm.ik_failures += 1
                print(f"Gesture {labels[frame]}, frame {frame}, {arm.name}: "
                      f"rejected after {attempts} attempts: {message}")
            plans.append((arm, row, candidate, clipped))
        # Commit the paired frame only when both arms have acceptable poses.
        if any(plan[2] is None for plan in plans):
            paired_rejections += 1
            for arm, row, candidate, clipped in plans:
                if candidate is not None:
                    arm.held_frames += 1
            continue
        # No simulation step or playback delay between these two updates.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        try:
            for arm, row, candidate, clipped in plans:
                for joint, value in zip(arm.joints, candidate):
                    p.resetJointState(arm.robot, joint, float(value))
        finally:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        for arm, row, candidate, clipped in plans:
            state = p.getLinkState(arm.robot, arm.tool_tip, computeForwardKinematics=True)
            position_error = np.linalg.norm(np.array(state[4]) - row[1:4]) * 1000
            orientation_error = np.degrees((
                Rotation.from_quat(row[4:8]).inv() * Rotation.from_quat(state[5])).magnitude())
            arm.results.append({"frame": frame, "position_error_mm": position_error,
                                "orientation_error_deg": orientation_error,
                                "limits_clipped": clipped, "joint_targets": candidate.copy()})
    print("\nPaired frames rejected (both arms held):", paired_rejections)
    print_summary(left, labels)
    print_summary(right, labels)
    print(f"Total runtime: {(time.perf_counter() - runtime_start)/60:.2f} minutes")
    print("Gripper angles are present in the CSVs; this model's grippers are not actuated here.")

    print_summary(left, labels)
    print_summary(right, right_labels)

# Print the per-gesture tables for both arms.
    print_gesture_results(left, labels)
    print_gesture_results(right, right_labels)



    while p.isConnected(0):
        time.sleep(1 / 60)
except KeyboardInterrupt:
    pass
finally:
    if p.isConnected(checker_client):
        p.disconnect(checker_client)
    if p.isConnected(0):
        p.disconnect(0)
