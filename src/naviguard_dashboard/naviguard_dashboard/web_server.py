"""
Lightweight HTTP Web Server for NAVIGUARD Operator Dashboard.

Provides REST endpoints and static file serving using standard library ThreadingHTTPServer.
Zero external pip dependencies.
"""

import os
import json
import urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Optional
import cv2
import numpy as np

from naviguard_dashboard.state_cache import StateCache
from naviguard_dashboard.coordinate_converter import MapCoordinateConverter
from naviguard_dashboard.goal_validator import GoalValidator


class NaviguardRequestHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests for dashboard UI and telemetry APIs."""

    state_cache: StateCache = None
    coord_converter: MapCoordinateConverter = None
    goal_validator: GoalValidator = None
    ros_bridge = None  # Reference to dashboard ROS node
    static_dir: str = ""
    _cached_basemap_bytes: Optional[bytes] = None


    def _safe_write(self, data: bytes) -> bool:
        """Safely write data to client socket, swallowing disconnect/broken pipe exceptions."""
        try:
            self.wfile.write(data)
            return True
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return False
        except Exception:
            return False

    def log_message(self, format, *args):
        """Suppress noisy access logging in console."""
        return

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        # 1. Static HTML
        if path == '/' or path == '/index.html':
            self._serve_file(os.path.join(self.static_dir, 'index.html'), 'text/html')
            return

        # 2. Telemetry Status Snapshot
        elif path == '/api/status':
            snapshot = self.state_cache.get_snapshot()
            data = json.dumps(snapshot).encode('utf-8')
            try:
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self._safe_write(data)
            except Exception:
                pass
            return

        # 3. Live Camera / Perception / VO Frames
        elif path == '/api/camera':
            frame_type = query.get('type', ['raw'])[0]
            jpeg_bytes = self.state_cache.get_jpeg_frame(frame_type)
            if not jpeg_bytes:
                # Dedicated distinct watermark frame per stream
                blank = np.zeros((240, 320, 3), dtype=np.uint8)
                topic_map = {
                    'raw': '/camera/image_raw',
                    'perception': '/perception/debug_image',
                    'segmentation': '/perception/segmentation',
                    'vo': '/visual_odometry/debug_image',
                    'chase': '/camera/chase_image',
                }
                topic = topic_map.get(frame_type, f'/{frame_type}')
                cv2.rectangle(blank, (4, 4), (316, 236), (30, 30, 30), 1)
                cv2.putText(blank, f"WAITING: {frame_type.upper()}", (25, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
                cv2.putText(blank, f"Topic: {topic}", (25, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 140), 1)
                cv2.putText(blank, "Awaiting publisher...", (25, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 180, 220), 1)
                _, enc = cv2.imencode('.jpg', blank)
                jpeg_bytes = enc.tobytes()

            try:
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(jpeg_bytes)))
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.end_headers()
                self._safe_write(jpeg_bytes)
            except Exception:
                pass
            return

        # 4. Map Image Rendering
        elif path == '/api/map_image':
            png_bytes = self.state_cache.map_png_bytes
            if not png_bytes:
                # 100x100 transparent 4-channel RGBA frame (Alpha=0) so it NEVER occludes basemap!
                blank = np.zeros((100, 100, 4), dtype=np.uint8)
                _, enc = cv2.imencode('.png', blank)
                png_bytes = enc.tobytes()

            try:
                self.send_response(200)
                self.send_header('Content-Type', 'image/png')
                self.send_header('Content-Length', str(len(png_bytes)))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.end_headers()
                self._safe_write(png_bytes)
            except Exception:
                pass
            return

        # 4b. RELLIS Realistic Basemap Rendering
        elif path == '/api/basemap':
            if NaviguardRequestHandler._cached_basemap_bytes is None:
                from naviguard_dashboard.basemap_generator import generate_rellis_basemap
                basemap_img = generate_rellis_basemap(600, 600, 0.05, -15.0, -15.0)
                _, enc = cv2.imencode('.png', basemap_img)
                NaviguardRequestHandler._cached_basemap_bytes = enc.tobytes()

            bmap_bytes = NaviguardRequestHandler._cached_basemap_bytes
            try:
                self.send_response(200)
                self.send_header('Content-Type', 'image/png')
                self.send_header('Content-Length', str(len(bmap_bytes)))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Cache-Control', 'public, max-age=3600')
                self.end_headers()
                self._safe_write(bmap_bytes)
            except Exception:
                pass
            return

        # 5. Goal Coordinate Conversion & Safety Validation
        elif path == '/api/validate_goal_click':

            try:
                u = float(query.get('u', [0])[0])
                v = float(query.get('v', [0])[0])
                cw = float(query.get('cw', [600])[0])
                ch = float(query.get('ch', [600])[0])

                # Update canvas dimensions in converter
                self.coord_converter.canvas_width = int(cw)
                self.coord_converter.canvas_height = int(ch)

                world_x, world_y = self.coord_converter.canvas_to_world(u, v)

                snap = self.state_cache.get_snapshot()
                rec_state = snap.get('recovery_state', 'NORMAL')
                # Operator goal selection is enabled unless the robot is halted in emergency FAILED_SAFE
                nav_online = (rec_state != 'FAILED_SAFE')
                has_localization = True

                is_valid, reason = self.goal_validator.validate_goal(
                    x=world_x,
                    y=world_y,
                    grid_data=self.state_cache.grid_data,
                    navigation_online=nav_online,
                    recovery_state=rec_state,
                    has_localization=has_localization
                )

                resp = {
                    'x': round(world_x, 2),
                    'y': round(world_y, 2),
                    'valid': is_valid,
                    'reason': reason
                }
            except Exception as e:
                resp = {'x': 0.0, 'y': 0.0, 'valid': False, 'reason': f'Conversion error: {e}'}

            data = json.dumps(resp).encode('utf-8')
            try:
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self._safe_write(data)
            except Exception:
                pass
            return

        else:
            try:
                self.send_response(404)
                self.end_headers()
            except Exception:
                pass

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_len = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_len) if content_len > 0 else b""

        if path in ['/api/goal', '/api/set_goal']:
            try:
                payload = json.loads(body.decode('utf-8'))
                x = float(payload.get('x', 0.0))
                y = float(payload.get('y', 0.0))
                yaw = float(payload.get('yaw', 0.0))

                snap = self.state_cache.get_snapshot()
                rec_state = snap.get('recovery_state', 'NORMAL')
                nav_online = (rec_state != 'FAILED_SAFE')
                has_localization = True

                is_valid, reason = self.goal_validator.validate_goal(
                    x=x,
                    y=y,
                    grid_data=self.state_cache.grid_data,
                    navigation_online=nav_online,
                    recovery_state=rec_state,
                    has_localization=has_localization
                )

                if not is_valid:
                    resp = {'success': False, 'message': reason}
                else:
                    success = self.ros_bridge.publish_navigation_goal(x, y, yaw)
                    if success:
                        self.state_cache.add_event("GOAL", f"Operator dispatched goal: X={x:.2f}m, Y={y:.2f}m")
                        resp = {'success': True, 'message': 'Goal successfully published to /goal_pose'}
                    else:
                        resp = {'success': False, 'message': 'ROS bridge failed to dispatch goal.'}
            except Exception as e:
                resp = {'success': False, 'message': f'Invalid goal request: {e}'}

            data = json.dumps(resp).encode('utf-8')
            try:
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self._safe_write(data)
            except Exception:
                pass
            return

        elif path == '/api/cancel_goal':
            success = self.ros_bridge.cancel_active_mission()
            self.state_cache.add_event("MISSION", "Operator requested mission cancellation.")
            resp = {'success': success, 'message': 'Cancellation dispatched'}
            data = json.dumps(resp).encode('utf-8')
            try:
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self._safe_write(data)
            except Exception:
                pass
            return

        elif path == '/api/fault/trigger':
            success = self.ros_bridge.trigger_recovery_fault()
            self.state_cache.add_event("FAULT", "Operator injected controlled recovery fault.")
            resp = {'success': success}
            data = json.dumps(resp).encode('utf-8')
            try:
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self._safe_write(data)
            except Exception:
                pass
            return

        elif path == '/api/fault/reset':
            success = self.ros_bridge.reset_recovery_budget()
            self.state_cache.add_event("RECOVERY", "Operator reset recovery budget.")
            resp = {'success': success}
            data = json.dumps(resp).encode('utf-8')
            try:
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self._safe_write(data)
            except Exception:
                pass
            return

        else:
            try:
                self.send_response(404)
                self.end_headers()
            except Exception:
                pass

    def _serve_file(self, filepath: str, content_type: str):
        if not os.path.exists(filepath):
            try:
                self.send_response(404)
                self.end_headers()
            except Exception:
                pass
            return
        with open(filepath, 'rb') as f:
            content = f.read()
        try:
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            self.end_headers()
            self._safe_write(content)
        except Exception:
            pass
