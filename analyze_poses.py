#!/usr/bin/env python
"""Analyze camera pose continuity and check for frame repetition issues."""
import numpy as np
import sys

def pose_enc_to_extrinsics(pose_enc):
    """Convert 9D pose encoding back to extrinsics.
    
    pose_enc format from DA3: [R_row0[:2], R_row1[:2], R_row2[:2], t] = [6 + 3] = 9
    Actually from the encoding: R[:2,:] (6 values) + t (3 values)
    """
    B, N = pose_enc.shape[0], pose_enc.shape[1]
    pose_enc = pose_enc.reshape(-1, 9)
    
    extrinsics = np.zeros((pose_enc.shape[0], 3, 4), dtype=np.float32)
    extrinsics[:, :2, :3] = pose_enc[:, :6].reshape(-1, 2, 3)
    extrinsics[:, :2, 3] = pose_enc[:, 6:9]
    extrinsics[:, 2, 3] = 1.0  # placeholder
    return extrinsics.reshape(B, N, 3, 4)


def extract_rotation_matrix(pose_enc):
    """Extract approximate rotation matrices from pose_enc.
    
    DA3 encodes: [R[:2,:].flatten(), t] = 9D
    We recover R[:2,:] and compute R[2,:] = R[0,:] x R[1,:] (cross product)
    Then orthonormalize.
    """
    B, N = pose_enc.shape[0], pose_enc.shape[1]
    R = np.zeros((B, N, 3, 3), dtype=np.float64)
    t = np.zeros((B, N, 3), dtype=np.float64)
    
    pe = pose_enc.reshape(-1, 9).astype(np.float64)
    R[:, :, :2, :] = pe[:, :6].reshape(B, N, 2, 3)
    t = pe[:, 6:9].reshape(B, N, 3)
    
    # Compute third row as cross product
    R[:, :, 2, :] = np.cross(R[:, :, 0, :], R[:, :, 1, :], axis=-1)
    
    # Orthonormalize using Gram-Schmidt
    for b in range(B):
        for n in range(N):
            r0 = R[b, n, 0]
            r1 = R[b, n, 1]
            # Normalize r0
            r0 = r0 / (np.linalg.norm(r0) + 1e-8)
            # Make r1 orthogonal to r0
            r1 = r1 - np.dot(r1, r0) * r0
            r1 = r1 / (np.linalg.norm(r1) + 1e-8)
            # Recompute r2
            r2 = np.cross(r0, r1)
            R[b, n, 0] = r0
            R[b, n, 1] = r1
            R[b, n, 2] = r2
    
    return R, t


def rotation_matrix_to_euler_angles(R):
    """Convert rotation matrix to Euler angles (ZYX convention).
    
    Returns roll, pitch, yaw in degrees.
    """
    # R is [B, N, 3, 3]
    sy = np.sqrt(R[..., 0, 0]**2 + R[..., 1, 0]**2)
    
    singular = sy < 1e-6
    
    roll = np.zeros(R.shape[:-2])
    pitch = np.zeros(R.shape[:-2])
    yaw = np.zeros(R.shape[:-2])
    
    not_singular = ~singular
    roll[not_singular] = np.arctan2(R[..., 2, 1][not_singular], R[..., 2, 2][not_singular])
    pitch[not_singular] = np.arctan2(-R[..., 2, 0][not_singular], sy[not_singular])
    yaw[not_singular] = np.arctan2(R[..., 1, 0][not_singular], R[..., 0, 0][not_singular])
    
    singular_mask = singular
    roll[singular_mask] = np.arctan2(-R[..., 1, 2][singular_mask], R[..., 1, 1][singular_mask])
    pitch[singular_mask] = np.arctan2(-R[..., 2, 0][singular_mask], sy[singular_mask])
    yaw[singular_mask] = 0.0
    
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def compute_angular_velocity(euler_angles, dt=1.0):
    """Compute angular velocity (first derivative) in deg/s."""
    # Unwrap angles to avoid discontinuities at ±180°
    unwrapped = np.unwrap(euler_angles, axis=0, period=360)
    angular_vel = np.diff(unwrapped, axis=0) / dt
    return angular_vel


def compute_angular_acceleration(euler_angles, dt=1.0):
    """Compute angular acceleration (second derivative) in deg/s²."""
    vel = compute_angular_velocity(euler_angles, dt)
    accel = np.diff(vel, axis=0) / dt
    return accel


def compute_translation_velocity(t, dt=1.0):
    """Compute translation velocity."""
    return np.diff(t, axis=0) / dt


def compute_translation_acceleration(t, dt=1.0):
    """Compute translation acceleration."""
    vel = compute_translation_velocity(t, dt)
    return np.diff(vel, axis=0) / dt


def detect_twisting(roll, window=20, threshold=5.0):
    """Detect persistent roll twisting (扭麻花).
    
    Check if roll angle monotonically increases/decreases over a window.
    """
    roll_unwrapped = np.unwrap(roll, period=360)
    roll_diff = np.diff(roll_unwrapped)
    
    # Count consecutive same-sign changes
    twist_count = 0
    max_twist = 0
    twist_starts = []
    
    for i in range(1, len(roll_diff)):
        if roll_diff[i] * roll_diff[i-1] > 0:  # Same direction
            twist_count += 1
            max_twist = max(max_twist, twist_count)
        else:
            if twist_count > window:
                twist_starts.append(i - twist_count)
            twist_count = 0
    
    if twist_count > window:
        twist_starts.append(len(roll_diff) - twist_count)
    
    return max_twist, twist_starts


def main():
    print("=" * 70)
    print("Camera Pose Continuity Analysis")
    print("=" * 70)
    
    # Load pose encoding
    pose_enc = np.load("output_full/results/pose_enc.npy")
    print(f"\n📦 Loaded pose_enc: {pose_enc.shape}")
    
    B, N = pose_enc.shape[0], pose_enc.shape[1]
    print(f"   Batch: {B}, Frames: {N}")
    
    # Check for frame repetition
    print(f"\n🔍 Checking for frame repetition...")
    # If frames were repeated, consecutive pose_enc values would be identical
    diff = np.abs(np.diff(pose_enc, axis=1))
    identical_frames = np.sum(diff < 1e-8, axis=(0, 2)) == 9
    if np.any(identical_frames):
        print(f"   ⚠️  Found {np.sum(identical_frames)} pairs of identical consecutive frames")
        print(f"   Indices: {np.where(identical_frames)[0]}")
    else:
        print(f"   ✅ No identical consecutive frames found")
    
    # Check pose_enc variation between consecutive frames
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    print(f"   Max pose_enc diff between frames: {max_diff:.6f}")
    print(f"   Mean pose_enc diff between frames: {mean_diff:.6f}")
    
    # Extract rotation and translation
    R, t = extract_rotation_matrix(pose_enc)
    roll, pitch, yaw = rotation_matrix_to_euler_angles(R)
    
    # Squeeze to [N]
    roll = roll[0]
    pitch = pitch[0]
    yaw = yaw[0]
    t = t[0]
    
    print(f"\n📐 Euler Angles (degrees):")
    print(f"   Roll:  [{roll[0]:.2f}, {roll[-1]:.2f}]  range=[{roll.min():.2f}, {roll.max():.2f}]")
    print(f"   Pitch: [{pitch[0]:.2f}, {pitch[-1]:.2f}]  range=[{pitch.min():.2f}, {pitch.max():.2f}]")
    print(f"   Yaw:   [{yaw[0]:.2f}, {yaw[-1]:.2f}]  range=[{yaw.min():.2f}, {yaw.max():.2f}]")
    
    print(f"\n📍 Translation:")
    print(f"   Start: {t[0]}")
    print(f"   End:   {t[-1]}")
    print(f"   Total displacement: {np.linalg.norm(t[-1] - t[0]):.4f}")
    
    # Angular velocity
    ang_vel = compute_angular_velocity(np.stack([roll, pitch, yaw], axis=-1))
    print(f"\n🔄 Angular Velocity (deg/frame):")
    print(f"   Roll:  mean={np.mean(np.abs(ang_vel[:, 0])):.4f}, max={np.max(np.abs(ang_vel[:, 0])):.4f}")
    print(f"   Pitch: mean={np.mean(np.abs(ang_vel[:, 1])):.4f}, max={np.max(np.abs(ang_vel[:, 1])):.4f}")
    print(f"   Yaw:   mean={np.mean(np.abs(ang_vel[:, 2])):.4f}, max={np.max(np.abs(ang_vel[:, 2])):.4f}")
    
    # Angular acceleration
    ang_acc = compute_angular_acceleration(np.stack([roll, pitch, yaw], axis=-1))
    print(f"\n⚡ Angular Acceleration (deg/frame²):")
    print(f"   Roll:  mean={np.mean(np.abs(ang_acc[:, 0])):.4f}, max={np.max(np.abs(ang_acc[:, 0])):.4f}")
    print(f"   Pitch: mean={np.mean(np.abs(ang_acc[:, 1])):.4f}, max={np.max(np.abs(ang_acc[:, 1])):.4f}")
    print(f"   Yaw:   mean={np.mean(np.abs(ang_acc[:, 2])):.4f}, max={np.max(np.abs(ang_acc[:, 2])):.4f}")
    
    # Translation velocity and acceleration
    trans_vel = compute_translation_velocity(t)
    trans_acc = compute_translation_acceleration(t)
    print(f"\n🏃 Translation Velocity (units/frame):")
    print(f"   mean={np.mean(np.linalg.norm(trans_vel, axis=1)):.6f}, max={np.max(np.linalg.norm(trans_vel, axis=1)):.6f}")
    print(f"\n🏃 Translation Acceleration (units/frame²):")
    print(f"   mean={np.mean(np.linalg.norm(trans_acc, axis=1)):.6f}, max={np.max(np.linalg.norm(trans_acc, axis=1)):.6f}")
    
    # Detect twisting
    max_twist, twist_starts = detect_twisting(roll, window=20, threshold=5.0)
    print(f"\n🌀 Roll Twisting Detection (扭麻花):")
    print(f"   Max consecutive same-direction roll changes: {max_twist}")
    if twist_starts:
        print(f"   ⚠️  Twisting detected at frames: {twist_starts}")
        for start in twist_starts[:5]:
            end = min(start + 50, len(roll))
            total_roll_change = np.abs(np.unwrap(roll[start:end], period=360)[-1] - np.unwrap(roll[start:end], period=360)[0])
            print(f"     Frame {start}-{end}: total roll change = {total_roll_change:.2f}°")
    else:
        print(f"   ✅ No persistent roll twisting detected")
    
    # Detect jumps (abnormal changes)
    ang_vel_magnitude = np.linalg.norm(ang_vel, axis=1)
    threshold_vel = np.mean(ang_vel_magnitude) + 3 * np.std(ang_vel_magnitude)
    jump_frames = np.where(ang_vel_magnitude > threshold_vel)[0]
    
    print(f"\n📊 Jump Detection (velocity > mean + 3σ = {threshold_vel:.4f}):")
    print(f"   Total jumps: {len(jump_frames)}")
    if len(jump_frames) > 0:
        for jf in jump_frames[:10]:
            print(f"     Frame {jf}: velocity = {ang_vel_magnitude[jf]:.4f}")
    
    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    
    issues = []
    if np.any(identical_frames):
        issues.append("⚠️  IDENTICAL CONSECUTIVE FRAMES DETECTED - possible frame repetition")
    if max_twist > 50:
        issues.append(f"⚠️  ROLL TWISTING: {max_twist} consecutive same-direction changes")
    if len(jump_frames) > N * 0.05:
        issues.append(f"⚠️  EXCESSIVE JUMPS: {len(jump_frames)} frames exceed velocity threshold")
    
    # Check if roll is monotonically increasing/decreasing over large range
    roll_range = np.abs(np.unwrap(roll, period=360)[-1] - np.unwrap(roll, period=360)[0])
    if roll_range > 180:
        issues.append(f"⚠️  LARGE ROLL CHANGE: {roll_range:.1f}° over sequence")
    
    if not issues:
        print("✅ No significant issues detected")
    else:
        for issue in issues:
            print(issue)
    
    # Save detailed analysis
    analysis = {
        "num_frames": N,
        "pose_enc_shape": list(pose_enc.shape),
        "identical_frames": int(np.sum(identical_frames)),
        "max_pose_enc_diff": float(max_diff),
        "mean_pose_enc_diff": float(mean_diff),
        "roll_range_deg": [float(roll.min()), float(roll.max())],
        "pitch_range_deg": [float(pitch.min()), float(pitch.max())],
        "yaw_range_deg": [float(yaw.min()), float(yaw.max())],
        "translation_start": t[0].tolist(),
        "translation_end": t[-1].tolist(),
        "total_displacement": float(np.linalg.norm(t[-1] - t[0])),
        "angular_velocity_max_deg": float(np.max(np.abs(ang_vel))),
        "angular_acceleration_max_deg": float(np.max(np.abs(ang_acc))),
        "translation_velocity_max": float(np.max(np.linalg.norm(trans_vel, axis=1))),
        "translation_acceleration_max": float(np.max(np.linalg.norm(trans_acc, axis=1))),
        "roll_twisting_max_consecutive": max_twist,
        "twisting_starts": [int(x) for x in twist_starts],
        "jump_frames": [int(x) for x in jump_frames],
        "jump_threshold": float(threshold_vel),
    }
    
    import json
    with open("output/pose_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n💾 Detailed analysis saved to output/pose_analysis.json")


if __name__ == "__main__":
    main()
