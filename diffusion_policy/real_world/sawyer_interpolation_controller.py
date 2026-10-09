"""
Sawyer/Intera drop-in replacement for RTDEInterpolationController.
"""

import os
import sys
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager

import numpy as np
import scipy.spatial.transform as st

from diffusion_policy.shared_memory.shared_memory_queue import SharedMemoryQueue, Empty
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator


HOME_JOINTS = {
    'right_j0':  0.0273525390625,
    'right_j1': -0.77991015625,
    'right_j2': -1.30016015625,
    'right_j3':  1.673435546875,
    'right_j4': -0.903478515625,
    'right_j5': -1.5015849609375,
    'right_j6': -0.1209189453125
}


def _ensure_ros_python_paths():
    ros_paths = [
        "/opt/ros/noetic/lib/python3/dist-packages",
        "/path/to/sawyer_robot_ws/devel/lib/python3/dist-packages",
    ]
    existing = os.environ.get("PYTHONPATH", "")
    parts = [p for p in existing.split(os.pathsep) if p]
    for path in ros_paths:
        if os.path.isdir(path):
            if path not in sys.path:
                sys.path.insert(0, path)
            if path not in parts:
                parts.insert(0, path)
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)
    os.environ["ROS_MASTER_URI"] = os.environ.get(
        "OBSTADIFF_ROS_MASTER_URI", "http://robot.local:11311")
    os.environ["ROS_HOSTNAME"] = os.environ.get(
        "OBSTADIFF_ROS_HOSTNAME", "workstation.local")


class Command(enum.Enum):
    STOP            = 0
    SERVOL          = 1
    SCHEDULE_WAYPOINT = 2
    ENABLE_ZERO_G   = 3
    DISABLE_ZERO_G  = 4


def _pose_to_matrix(pose6d):
    pos    = pose6d[:3]
    rotvec = pose6d[3:6]
    R = st.Rotation.from_rotvec(rotvec).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = pos
    return T


def _matrix_to_pose(T):
    pos    = T[:3, 3]
    rotvec = st.Rotation.from_matrix(T[:3, :3]).as_rotvec()
    return np.concatenate([pos, rotvec])


class SawyerInterpolationController(mp.Process):

    def __init__(self,
                 shm_manager: SharedMemoryManager,
                 frequency=100,
                 max_pos_speed=0.25,
                 max_rot_speed=0.6,
                 launch_timeout=15,
                 joints_init=None,
                 joints_init_speed=0.1,
                 verbose=False,
                 receive_keys=None,
                 get_max_k=128):

        assert 0 < frequency <= 500
        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (7,)

        super().__init__(name="SawyerInterpolationController")

        self.frequency        = frequency
        self.max_pos_speed    = max_pos_speed
        self.max_rot_speed    = max_rot_speed
        self.launch_timeout   = launch_timeout
        self.joints_init      = joints_init
        self.joints_init_speed = joints_init_speed
        self.verbose          = verbose

        # input queue
        example = {
            'cmd':         Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'duration':    0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # ring buffer
        if receive_keys is None:
            receive_keys = [
                'ActualTCPPose',
                'ActualTCPSpeed',
                'ActualQ',
                'ActualQd',
                'TargetTCPPose',
            ]
        self.receive_keys = receive_keys

        example_state = {
            'ActualTCPPose':           np.zeros(6,  dtype=np.float64),
            'ActualTCPSpeed':          np.zeros(6,  dtype=np.float64),
            'ActualQ':                 np.zeros(7,  dtype=np.float64),
            'ActualQd':                np.zeros(7,  dtype=np.float64),
            'TargetTCPPose':           np.zeros(6,  dtype=np.float64),
            'robot_receive_timestamp': time.time()
        }
        example_state = {k: v for k, v in example_state.items()
                         if k in receive_keys or k == 'robot_receive_timestamp'}

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example_state,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event  = mp.Event()
        self.input_queue  = input_queue
        self.ring_buffer  = ring_buffer

    # ── launch / stop ──────────────────────────────────────────────────────

    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        if self.is_alive():
            try:
                self.input_queue.put({
                    'cmd': Command.STOP.value,
                    'target_pose': np.zeros(6),
                    'duration': 0.0,
                    'target_time': 0.0
                })
            except Exception:
                pass
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive(), "SawyerController process died on startup"

    def stop_wait(self):
        self.join(timeout=3.0)
        if self.is_alive():
            self.terminate()
            self.join(timeout=2.0)
        if self.is_alive():
            self.kill()
            self.join(timeout=1.0)

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ── command methods ────────────────────────────────────────────────────

    def servoL(self, pose, duration=0.1):
        assert self.is_alive()
        assert duration >= (1 / self.frequency)
        pose = np.array(pose, dtype=np.float64)
        assert pose.shape == (6,)
        self.input_queue.put({
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration,
            'target_time': 0.0
        })

    def schedule_waypoint(self, pose, target_time):
        assert target_time > time.time()
        pose = np.array(pose, dtype=np.float64)
        assert pose.shape == (6,)
        self.input_queue.put({
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'duration': 0.0,
            'target_time': target_time
        })

    def enable_zero_g(self):
        """Enable gravity compensation — arm floats freely."""
        self.input_queue.put({
            'cmd': Command.ENABLE_ZERO_G.value,
            'target_pose': np.zeros(6),
            'duration': 0.0,
            'target_time': 0.0
        })

    def disable_zero_g(self):
        """Disable gravity compensation and return to home position."""
        self.input_queue.put({
            'cmd': Command.DISABLE_ZERO_G.value,
            'target_pose': np.zeros(6),
            'duration': 0.0,
            'target_time': 0.0
        })

    # ── receive APIs ───────────────────────────────────────────────────────

    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    # ── main loop ──────────────────────────────────────────────────────────

    def run(self):
        import traceback
        # redirect stderr to a log file so we can see errors
        sys.stderr = open('/tmp/sawyer_controller_stderr.txt', 'w')
        sys.stdout = open('/tmp/sawyer_controller_stdout.txt', 'w')
        _ensure_ros_python_paths()
        # Import ROS here so init happens only in subprocess
        import os
        import socket
        from urllib.parse import urlparse
        master_uri = os.environ.get('OBSTADIFF_ROS_MASTER_URI', 'http://robot.local:11311')
        os.environ['ROS_MASTER_URI'] = master_uri
        # Resolve the master hostname to IP so subprocess can find it
        try:
            parsed = urlparse(master_uri)
            port = parsed.port or 11311
            ip = socket.gethostbyname(parsed.hostname)
            os.environ['ROS_MASTER_URI'] = f'http://{ip}:{port}'
            print(f"[SawyerController] ROS master at {ip}:{port}")
        except Exception:
            pass
        import rospy
        import intera_interface
        from intera_interface import Limb
        from intera_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest
        from geometry_msgs.msg import PoseStamped
        from tf.transformations import quaternion_from_matrix

        rospy.init_node('sawyer_diffpol_controller', anonymous=True,
                        disable_signals=True)

        rs = intera_interface.RobotEnable(intera_interface.CHECK_VERSION)
        rs.enable()

        limb    = Limb('right')
        #gripper = intera_interface.Gripper('right_gripper')
        #gripper.calibrate()

        ik_service = rospy.ServiceProxy(
            'ExternalTools/right/PositionKinematicsNode/IKService',
            SolvePositionIK)

        zero_g_active = False

        def get_current_pose():
            ep   = limb.endpoint_pose()
            pos  = ep['position']
            quat = ep['orientation']
            R    = st.Rotation.from_quat(
                       [quat.x, quat.y, quat.z, quat.w]).as_matrix()
            T    = np.eye(4)
            T[:3, :3] = R
            T[:3, 3]  = [pos.x, pos.y, pos.z]
            return _matrix_to_pose(T)

        def move_to_pose(pose6d):
            T  = _pose_to_matrix(pose6d)
            ps = PoseStamped()
            ps.header.frame_id = 'base'
            ps.header.stamp    = rospy.Time.now()
            ps.pose.position.x = T[0, 3]
            ps.pose.position.y = T[1, 3]
            ps.pose.position.z = T[2, 3]
            q = quaternion_from_matrix(T)
            ps.pose.orientation.x = q[0]
            ps.pose.orientation.y = q[1]
            ps.pose.orientation.z = q[2]
            ps.pose.orientation.w = q[3]
            req = SolvePositionIKRequest()
            req.tip_names.append('right_hand')
            req.pose_stamp.append(ps)
            resp = ik_service(req)
            if resp.result_type[0] > 0:
                jnames  = resp.joints[0].name
                jangles = resp.joints[0].position
                limb.set_joint_positions(dict(zip(jnames, jangles)))
                return True
            return False

        try:
            # optional init move
            if self.joints_init is not None:
                limb.set_joint_position_speed(self.joints_init_speed)
                limb.move_to_joint_positions(
                    dict(zip(limb.joint_names(), self.joints_init)),
                    timeout=15.0)
                if self.verbose:
                    print("[SawyerController] Reached init joints")

            limb.set_joint_position_speed(0.3)

            curr_pose   = get_current_pose()
            curr_t      = time.monotonic()
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t], poses=[curr_pose])
            last_waypoint_time = curr_t
            dt           = 1.0 / self.frequency
            iter_idx     = 0
            keep_running = True

            while keep_running and not rospy.is_shutdown():
                t_start = time.monotonic()
                t_now   = time.monotonic()

                # only send pose commands when not in zero-G
                if not zero_g_active:
                    pose_command = pose_interp(t_now)
                    move_to_pose(pose_command)
                else:
                    pose_command = get_current_pose()

                # read robot state
                ja  = limb.joint_angles()
                jv  = limb.joint_velocities()
                ep  = limb.endpoint_pose()
                ev  = limb.endpoint_velocity()

                pos  = ep['position']
                quat = ep['orientation']
                R    = st.Rotation.from_quat(
                           [quat.x, quat.y, quat.z, quat.w])
                actual_pose  = np.array([pos.x, pos.y, pos.z,
                                         *R.as_rotvec()])
                lv = ev['linear']
                av = ev['angular']
                actual_speed = np.array([lv.x, lv.y, lv.z,
                                         av.x, av.y, av.z])
                jnames   = limb.joint_names()
                actual_q  = np.array([ja[n] for n in jnames])
                actual_qd = np.array([jv[n] for n in jnames])

                state = {
                    'ActualTCPPose':           actual_pose,
                    'ActualTCPSpeed':          actual_speed,
                    'ActualQ':                 actual_q,
                    'ActualQd':                actual_qd,
                    'TargetTCPPose':           pose_command,
                    'robot_receive_timestamp': time.time()
                }
                state = {k: v for k, v in state.items()
                         if k in self.receive_keys
                         or k == 'robot_receive_timestamp'}
                self.ring_buffer.put(state)

                # fetch commands
                try:
                    commands = self.input_queue.get_all()
                    n_cmd    = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                for i in range(n_cmd):
                    cmd = commands['cmd'][i]

                    if cmd == Command.STOP.value:
                        keep_running = False
                        break

                    elif cmd == Command.ENABLE_ZERO_G.value:
                        print("[SawyerController] Enabling zero-G")
                        zero_torques = dict(zip(limb.joint_names(), [0.0]*7))
                        limb.set_joint_torques(zero_torques)
                        zero_g_active = True

                    elif cmd == Command.DISABLE_ZERO_G.value:
                        print("[SawyerController] Disabling zero-G, going home")
                        zero_g_active = False
                        limb.set_joint_position_speed(0.1)
                        limb.move_to_joint_positions(HOME_JOINTS, timeout=15.0)
                        # reset interpolator to home pose
                        curr_pose   = get_current_pose()
                        curr_t      = time.monotonic()
                        pose_interp = PoseTrajectoryInterpolator(
                            times=[curr_t], poses=[curr_pose])
                        last_waypoint_time = curr_t

                    elif cmd == Command.SERVOL.value:
                        target_pose = commands['target_pose'][i]
                        duration    = float(commands['duration'][i])
                        curr_time   = t_now + dt
                        t_insert    = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed
                        )
                        last_waypoint_time = t_insert

                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = commands['target_pose'][i]
                        target_time = float(commands['target_time'][i])
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time   = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time

                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                elapsed    = time.monotonic() - t_start
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        finally:
            limb.set_joint_velocities(
                dict(zip(limb.joint_names(), [0.0] * 7)))
            self.ready_event.set()
            if self.verbose:
                print("[SawyerController] Stopped.")
