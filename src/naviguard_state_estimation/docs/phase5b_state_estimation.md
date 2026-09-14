# NAVIGUARD Phase 5B: Multi-Sensor Metric State Estimation

## 1. Package Architecture

The `naviguard_state_estimation` package provides a baseline metric state estimator for the NAVIGUARD 4-wheel differential-drive outdoor UGV. It mathematically fuses three distinct sensing modalities:
1. **Wheel Odometry (`/odom`)**: Metric linear velocity $v_x$, yaw rate $\omega_z$, and reference frame pose.
2. **IMU (`/imu`)**: High-frequency angular yaw rate $\omega_z$ and body linear acceleration $a_x$.
3. **Visual Motion Telemetry (`/visual_odometry/telemetry`)**: Geometric visual translation unit direction $\hat{\mathbf{t}}$ and optical frame yaw rate from Phase 4 monocular geometric visual odometry.

### Modules

- `state_model.py`: Defines the 5-DOF state vector $\mathbf{x} = [x, y, \theta, v_x, \omega_z]^T$ in the `odom` frame, non-linear unicycle kinematic propagation, and discrete-time state transition Jacobian $F_x$.
- `estimator.py`: Implements an Extended Kalman Filter (EKF) with Mahalanobis outlier gating (1D for IMU/visual yaw, 2D for wheel velocity and visual scaled updates) and Joseph-form covariance updates $P = (I - KH)P(I - KH)^T + KRK^T$ ensuring strict symmetry and positive-definiteness.
- `measurement_buffer.py`: Chronological sliding-window buffer providing timestamp-sorted storage, nearest-neighbor lookups, and windowed numerical trapezoidal integration of wheel displacement.
- `imu_processor.py`: Gates raw 6-axis IMU measurements, enforces angular velocity and linear acceleration thresholds, and avoids false orientation assumptions (`has_orientation: False`).
- `wheel_odom_processor.py`: Extracts linear and angular velocities, validates frame consistency (`odom` $\to$ `base_link`), enforces speed limits and non-holonomic lateral constraints.
- `visual_measurement_adapter.py`: Converts optical frame coordinates into `base_link` coordinates and evaluates local metric scale using wheel odometry constraints.
- `consistency_checker.py`: Tracks physical cross-sensor residuals across visual, wheel, and IMU modalities to detect wheel slip, visual tracking failure, or sensor divergence.
- `diagnostics.py`: Generates structured `diagnostic_msgs/msg/DiagnosticArray` messages on `/state_estimation/diagnostics` and optional trajectory CSV logs.
- `state_estimation_node.py`: ROS 2 Jazzy node coordinating buffers, filter prediction, measurement updates, and timer-driven publishing of `/state_estimation/odom` and `/state_estimation/diagnostics`.

---

## 2. Monocular Scale Observation Architecture

### The Monocular Scale Ambiguity
Monocular camera visual motion estimation via epipolar geometry (`recoverPose`) recovers the translation direction up to an arbitrary scale factor:
$$\|\hat{\mathbf{t}}_{opt}\| = 1$$
No metric distance can be inferred from monocular images alone without external priors.

### Coordinate Transformation
From the URDF definition, the camera is mounted with optical orientation:
- Optical $+Z$ corresponds to robot forward ($+X_{base}$)
- Optical $+X$ corresponds to robot right ($-Y_{base}$)
- Optical $+Y$ corresponds to robot down ($-Z_{base}$)
- Optical yaw rotation (around $+Y_{opt}$, down) corresponds to negative yaw rotation in base_link ($+Z_{base}$, up):
$$\omega_{z,base} = -\frac{\text{rot\_yaw\_opt}}{\Delta t}$$

### Metric Scale Observation & Gating
Local metric scale is observed only through wheel-motion constraints when the following criteria are simultaneously met:
1. **Geometry Validity**: `geom_is_valid == True` (cheirality check satisfied, sufficient parallax).
2. **Inlier Count**: `geom_num_inliers >= min_inliers` (default: 15).
3. **Sufficient Wheel Motion**: Wheel displacement over the visual interval $|d_{wheel}| \ge d_{min}$ (default: 0.02 m) and wheel speed $|v_{wheel}| \ge v_{min}$ (default: 0.05 m/s).
4. **Directional Alignment**: Angle between visual translation vector and forward heading:
$$\theta_{err} = \left|\arctan2(t_{base,y}, t_{base,x})\right| \le \theta_{max} \quad (\text{default: } 45^\circ)$$

When all conditions hold:
$$\text{scale}_{obs} = \frac{|d_{wheel}|}{\sqrt{t_{base,x}^2 + t_{base,y}^2}}$$
$$v_{x,metric} = \frac{\text{scale}_{obs} \cdot t_{base,x}}{\Delta t}$$
The filter applies a 2-DOF update $[v_x, \omega_z]^T$. If any gating condition fails, scale observation is marked unobserved/rejected, and the filter falls back to a 1-DOF update on visual yaw rate $\omega_z$ only.

---

## 3. Kinematic State Model and EKF Formulation

### State Vector
$$\mathbf{x} = \begin{bmatrix} x \\ y \\ \theta \\ v_x \\ \omega_z \end{bmatrix} \in \mathbb{R}^5$$

### Non-Linear Unicycle Kinematic Propagation
For time step $\Delta t$, the state propagates as:
$$\theta_{mid} = \theta_k + \omega_{z,k} \frac{\Delta t}{2}$$
$$x_{k+1} = x_k + v_{x,k} \cos(\theta_{mid}) \Delta t$$
$$y_{k+1} = y_k + v_{x,k} \sin(\theta_{mid}) \Delta t$$
$$\theta_{k+1} = \text{wrap}(\theta_k + \omega_{z,k} \Delta t)$$
$$v_{x,k+1} = v_{x,k}$$
$$\omega_{z,k+1} = \omega_{z,k}$$

### Discrete-Time State Transition Jacobian $F_x = \frac{\partial f}{\partial \mathbf{x}}$
$$F_x = \begin{bmatrix}
1 & 0 & -v_x \sin(\theta_{mid}) \Delta t & \cos(\theta_{mid}) \Delta t & -\frac{1}{2} v_x \sin(\theta_{mid}) \Delta t^2 \\
0 & 1 & v_x \cos(\theta_{mid}) \Delta t & \sin(\theta_{mid}) \Delta t & \frac{1}{2} v_x \cos(\theta_{mid}) \Delta t^2 \\
0 & 0 & 1 & 0 & \Delta t \\
0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 1
\end{bmatrix}$$

### Covariance Propagation
$$P_{k+1|k} = F_x P_{k|k} F_x^T + Q \Delta t$$
$$Q = \text{diag}(q_{pos}, q_{pos}, q_\theta, q_{vx}, q_{\omega z})$$

### Measurement Updates and Mahalanobis Gating
For measurement $\mathbf{z}$ with measurement model $\mathbf{h}(\mathbf{x})$ and covariance $R$:
$$\mathbf{y} = \mathbf{z} - H \mathbf{x}$$
$$S = H P H^T + R$$
Mahalanobis distance test:
$$d_M^2 = \mathbf{y}^T S^{-1} \mathbf{y}$$
- 1D updates: $d_M^2 \le 9.0$ ($3\sigma$)
- 2D updates: $d_M^2 \le 13.8$ ($3.7\sigma$)

If $d_M^2$ exceeds threshold, the measurement is flagged as an outlier and rejected.
When accepted, the Kalman gain is:
$$K = P H^T S^{-1}$$
$$\mathbf{x} \leftarrow \mathbf{x} + K \mathbf{y}$$
$$P \leftarrow (I - KH) P (I - KH)^T + K R K^T$$

---

## 4. Cross-Sensor Consistency Monitoring

The consistency checker computes physical residuals between independent sensor channels:
1. **Visual vs Wheel**:
   - Yaw rate residual: $r_{wz} = |\omega_{z,vis} - \omega_{z,wheel}|$
   - Motion direction residual: $\theta_{err} = \text{angle}(\hat{\mathbf{t}}_{base}, [1, 0]^T)$
2. **Visual vs IMU**:
   - Yaw rate residual: $r_{wz} = |\omega_{z,vis} - \omega_{z,imu}|$
3. **Wheel vs IMU**:
   - Yaw rate residual: $r_{wz} = |\omega_{z,wheel} - \omega_{z,imu}|$
   - Acceleration residual: $r_{ax} = \left|\frac{\Delta v_{x,wheel}}{\Delta t} - a_{x,imu}\right|$

Divergences exceeding physical bounds trigger warning diagnostics (`DiagnosticStatus.WARN`) without destabilizing the state estimator.

---

## 5. Topics and TF Strategy

### Subscribed Topics
- `/odom` (`nav_msgs/msg/Odometry`, RELIABLE QoS)
- `/imu` (`sensor_msgs/msg/Imu`, BEST_EFFORT QoS)
- `/visual_odometry/telemetry` (`diagnostic_msgs/msg/DiagnosticArray`, RELIABLE QoS)

### Published Topics
- `/state_estimation/odom` (`nav_msgs/msg/Odometry`, 20 Hz): Fused metric state estimate in `odom` frame with full 6x6 pose and twist covariances.
- `/state_estimation/diagnostics` (`diagnostic_msgs/msg/DiagnosticArray`, 20 Hz): Multi-status diagnostic array detailing state, sensor updates, scale observations, and consistency residuals.

### TF Strategy
To avoid conflicting transforms on `/tf`, `publish_tf` is set to `false` by default because Gazebo's differential-drive plugin already broadcasts the ground-truth `odom` $\to$ `base_link` transform.
