import sys
import ctypes
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time
import os
from datetime import datetime
import math


class AirPaintApp:
    def __init__(self):
        self.WINDOW_WIDTH = 640
        self.WINDOW_HEIGHT = 480
        self.BUTTON_HEIGHT = 50
        self.BUTTON_MARGIN = 5

        self.colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
        self.color_names = ["R", "G", "B", "Y"]
        self.current_color_index = 0

        self.brush_sizes = [2, 5, 8, 12]
        self.brush_size_names = ["XS", "S", "M", "L"]
        self.current_brush_size_index = 1

        self.tools = {
            "eraser": True,
            "undo_redo": True,
            "save": True,
        }
        self.current_tool = "brush"

        self.last_pos = None
        self.drawing_mode = False

        self.last_tool_change = 0
        self.hover_button = None

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            ctypes.windll.user32.MessageBoxW(
                0,
                "Error: No supported webcam detected.",
                "AirCanvas",
                0x10,
            )
            sys.exit(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.WINDOW_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.WINDOW_HEIGHT)

        self.canvas_history = []
        self.history_index = -1
        self.max_history = 20

        self.prev_drawing = False
        self.prev_pinch = False

        self.create_paint_window()

    def create_paint_window(self):
        """Create the paint canvas"""
        self.paint_window = (
            np.ones((self.WINDOW_HEIGHT, self.WINDOW_WIDTH, 3), dtype=np.uint8) * 255
        )
        self.save_to_history()

    def save_to_history(self):
        """Save current canvas state to history for undo/redo"""
        if not self.tools["undo_redo"]:
            return

        if self.history_index < len(self.canvas_history) - 1:
            self.canvas_history = self.canvas_history[: self.history_index + 1]

        self.canvas_history.append(self.paint_window.copy())

        if len(self.canvas_history) > self.max_history:
            self.canvas_history.pop(0)
        else:
            self.history_index += 1

    def undo(self):
        """Undo last action"""
        if self.history_index > 0:
            self.history_index -= 1
            self.paint_window = self.canvas_history[self.history_index].copy()

    def redo(self):
        """Redo last undone action"""
        if self.history_index < len(self.canvas_history) - 1:
            self.history_index += 1
            self.paint_window = self.canvas_history[self.history_index].copy()

    def draw_rounded_rectangle(self, img, pt1, pt2, color, thickness=-1, radius=10):
        """Draw rounded rectangle"""
        x1, y1 = pt1
        x2, y2 = pt2

        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)

        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)

    def create_gradient_background(self, frame):
        """Create gradient background for better visual appeal"""
        height, width = frame.shape[:2]
        gradient = np.zeros((height, width, 3), dtype=np.uint8)

        for i in range(height):
            ratio = i / height
            gradient[i, :] = [
                int(20 + ratio * 10),
                int(25 + ratio * 15),
                int(35 + ratio * 20),
            ]

        return cv2.addWeighted(frame, 0.7, gradient, 0.3, 0)

    def draw_ui_elements(self, frame):
        """Draw all UI elements on the frame"""
        frame[: self.BUTTON_HEIGHT + 10] = self.create_gradient_background(
            frame[: self.BUTTON_HEIGHT + 10]
        )

        # Brush sizes
        brush_width = 40
        start_x = 10
        for i, size_name in enumerate(self.brush_size_names):
            x1 = start_x + i * (brush_width + self.BUTTON_MARGIN)
            x2 = x1 + brush_width
            size_color = (
                (100, 150, 200) if i == self.current_brush_size_index else (80, 80, 80)
            )
            self.draw_rounded_rectangle(frame, (x1, 5), (x2, 45), size_color, -1, 6)
            cv2.putText(
                frame,
                size_name,
                (x1 + 5, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        # Colors
        color_width = 40
        start_x = start_x + 4 * (brush_width + self.BUTTON_MARGIN)
        for i, (color, name) in enumerate(zip(self.colors, self.color_names)):
            x1 = start_x + i * (color_width + self.BUTTON_MARGIN)
            x2 = x1 + color_width

            button_color = (
                color
                if i == self.current_color_index
                else tuple(int(c * 0.7) for c in color)
            )
            self.draw_rounded_rectangle(frame, (x1, 5), (x2, 45), button_color, -1, 8)

            text_color = (255, 255, 255) if name != "YELLOW" else (0, 0, 0)
            cv2.putText(
                frame,
                name[:3],
                (x1 + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                text_color,
                1,
            )

        # Clear
        other_width = 40
        start_x = start_x + 4 * (color_width + self.BUTTON_MARGIN)
        clear_x = start_x
        self.draw_rounded_rectangle(
            frame, (clear_x, 5), (clear_x + other_width, 45), (220, 220, 220), -1, 8
        )
        cv2.putText(
            frame,
            "CLR",
            (clear_x + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
        )

        # Erase
        erase_x = clear_x + other_width + self.BUTTON_MARGIN
        if self.tools["eraser"]:
            eraser_color = (
                (200, 200, 200) if self.current_tool == "eraser" else (150, 150, 150)
            )
            self.draw_rounded_rectangle(
                frame,
                (erase_x, 5),
                (erase_x + other_width, 45),
                eraser_color,
                -1,
                8,
            )
            cv2.putText(
                frame,
                "ERS",
                (erase_x + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
            )

        # Undo
        undo_x = erase_x + other_width + self.BUTTON_MARGIN
        if self.tools["undo_redo"]:
            undo_color = (180, 180, 180) if self.history_index > 0 else (120, 120, 120)
            self.draw_rounded_rectangle(
                frame, (undo_x, 5), (undo_x + other_width, 45), undo_color, -1, 6
            )
            cv2.putText(
                frame,
                "UND",
                (undo_x + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
            )

        # Redo
        redo_x = undo_x + other_width + self.BUTTON_MARGIN
        if self.tools["undo_redo"]:
            redo_color = (
                (180, 180, 180)
                if self.history_index < len(self.canvas_history) - 1
                else (120, 120, 120)
            )
            self.draw_rounded_rectangle(
                frame, (redo_x, 5), (redo_x + other_width, 45), redo_color, -1, 6
            )
            cv2.putText(
                frame,
                "RDO",
                (redo_x + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
            )

        # Save
        save_x = redo_x + other_width + self.BUTTON_MARGIN
        if self.tools["save"]:
            self.draw_rounded_rectangle(
                frame, (save_x, 5), (save_x + other_width, 45), (100, 200, 100), -1, 8
            )
            cv2.putText(
                frame,
                "SAV",
                (save_x + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

        status_y = self.WINDOW_HEIGHT - 10
        status_text = f"Tool: {self.current_tool.upper()}  |  Size: {self.brush_sizes[self.current_brush_size_index]}px"
        if self.current_tool == "brush":
            status_text += f"  |  Color: {self.color_names[self.current_color_index]}"

        cv2.putText(
            frame,
            status_text,
            (10, status_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

        if time.time() - self.last_tool_change < 1.5:
            cv2.putText(
                frame,
                f"Tool: {self.current_tool.upper()}",
                (frame.shape[1] // 2 - 50, frame.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

    def detect_button_click(self, point):
        """Detect which button was clicked"""
        x, y = point

        if y > self.BUTTON_HEIGHT + 10:
            return None

        # Brush sizes
        brush_width = 40
        start_x = 10
        for i in range(len(self.brush_size_names)):
            x1 = start_x + i * (brush_width + self.BUTTON_MARGIN)
            if x1 <= x <= x1 + brush_width:
                return f"brush_size_{i}"

        # Colors
        color_width = 40
        start_x = start_x + 4 * (brush_width + self.BUTTON_MARGIN)
        for i in range(len(self.colors)):
            x1 = start_x + i * (color_width + self.BUTTON_MARGIN)
            if x1 <= x <= x1 + color_width:
                return f"color_{i}"

        # Clear
        other_width = 40
        start_x = start_x + 4 * (color_width + self.BUTTON_MARGIN)
        if start_x <= x <= start_x + other_width:
            return "clear"

        # Erase
        start_x += other_width + self.BUTTON_MARGIN
        if self.tools["eraser"] and start_x <= x <= start_x + other_width:
            return "eraser"

        # Undo
        start_x += other_width + self.BUTTON_MARGIN
        if self.tools["undo_redo"] and start_x <= x <= start_x + other_width:
            return "undo"

        # Redo
        start_x += other_width + self.BUTTON_MARGIN
        if self.tools["undo_redo"] and start_x <= x <= start_x + other_width:
            return "redo"

        # Save
        start_x += other_width + self.BUTTON_MARGIN
        if self.tools["save"] and start_x <= x <= start_x + other_width:
            return "save"

        return None

    def handle_button_action(self, action):
        """Handle button actions"""
        if action == "clear":
            self.paint_window[:] = 255
            self.save_to_history()

        elif action.startswith("color_"):
            color_index = int(action.split("_")[1])
            self.current_color_index = color_index
            self.current_tool = "brush"
            self.last_tool_change = time.time()

        elif action == "eraser":
            self.current_tool = "eraser" if self.current_tool != "eraser" else "brush"
            self.last_tool_change = time.time()

        elif action == "undo":
            self.undo()

        elif action == "redo":
            self.redo()

        elif action == "save":
            self.save_drawing()

        elif action.startswith("brush_size_"):
            size_index = int(action.split("_")[2])
            self.current_brush_size_index = size_index

    def save_drawing(self):
        """Save the current drawing"""
        if not os.path.exists("drawings"):
            os.makedirs("drawings")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"drawings/air_paint_{timestamp}.png"
        cv2.imwrite(filename, self.paint_window)
        print(f"Drawing saved as {filename}")

    def detect_gestures(self, landmarks, frame_shape):
        """Enhanced gesture detection"""
        if len(landmarks) < 21:
            return None, False, False

        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        index_mcp = landmarks[5]
        index_pip = landmarks[6]
        index_dip = landmarks[7]

        middle_mcp = landmarks[9]
        middle_pip = landmarks[10]
        middle_dip = landmarks[11]

        ring_mcp = landmarks[13]
        ring_pip = landmarks[14]
        ring_dip = landmarks[15]

        pinky_mcp = landmarks[17]
        pinky_pip = landmarks[18]
        pinky_dip = landmarks[19]

        index_extended = index_tip[1] < index_dip[1] < index_pip[1] < index_mcp[1]
        middle_extended = middle_tip[1] < middle_dip[1] < middle_pip[1] < middle_mcp[1]
        ring_extended = ring_tip[1] < ring_dip[1] < ring_pip[1] < ring_mcp[1]
        pinky_extended = pinky_tip[1] < pinky_dip[1] < pinky_pip[1] < pinky_mcp[1]

        fingers_closed = (
            not middle_extended and not ring_extended and not pinky_extended
        )

        thumb_index_dist = math.hypot(
            thumb_tip[0] - index_tip[0], thumb_tip[1] - index_tip[1]
        )

        pinch = thumb_index_dist < 25
        pinch_event = pinch and not self.prev_pinch
        self.prev_pinch = pinch

        is_drawing = False
        finger_pos = None

        if index_extended and fingers_closed:
            finger_pos = index_tip
            if thumb_index_dist > 40:
                is_drawing = True

        return finger_pos, is_drawing, pinch_event

    def show_settings_screen(self):
        """Show tool selection screen"""
        settings_window = np.ones((400, 600, 3), dtype=np.uint8) * 240

        cv2.putText(
            settings_window,
            "AIR PAINT - TOOL SETTINGS",
            (120, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (50, 50, 50),
            2,
        )

        cv2.putText(
            settings_window,
            "Enable/Disable tools by clicking:",
            (150, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (100, 100, 100),
            1,
        )

        y_start = 130
        tool_names = [
            "Eraser Tool",
            "Undo/Redo",
            "Save Function",
        ]
        tool_keys = ["eraser", "undo_redo", "save"]

        self.tool_buttons = []

        for i, (name, key) in enumerate(zip(tool_names, tool_keys)):
            y = y_start + i * 45

            checkbox_color = (100, 200, 100) if self.tools[key] else (200, 100, 100)
            thickness = -1 if self.tools[key] else 2
            cv2.rectangle(
                settings_window, (100, y - 15), (125, y + 10), checkbox_color, thickness
            )
            cv2.rectangle(settings_window, (100, y - 15), (125, y + 10), (0, 0, 0), 2)

            cv2.putText(
                settings_window,
                name,
                (140, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (50, 50, 50),
                2,
            )

            self.tool_buttons.append((100, y - 15, 125, y + 10, key))

        self.draw_rounded_rectangle(
            settings_window, (250, 320), (350, 360), (100, 200, 100), -1, 8
        )
        cv2.rectangle(settings_window, (250, 320), (350, 360), (0, 0, 0), 2)
        cv2.putText(
            settings_window,
            "START",
            (270, 345),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        return settings_window

    def handle_settings_click(self, x, y):
        """Handle clicks on settings screen"""
        for x1, y1, x2, y2, key in self.tool_buttons:
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.tools[key] = not self.tools[key]
                return False

        if 250 <= x <= 350 and 320 <= y <= 360:
            return True

        return False

    def settings_mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for settings screen"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.handle_settings_click(x, y):
                cv2.destroyWindow("Settings")
                self.run_main_application()

    def run_settings(self):
        """Run the settings screen"""
        cv2.namedWindow("Settings")
        cv2.setMouseCallback("Settings", self.settings_mouse_callback)

        while True:
            settings_frame = self.show_settings_screen()
            cv2.imshow("Settings", settings_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == 13:
                cv2.destroyWindow("Settings")
                self.run_main_application()
                break

        cv2.destroyAllWindows()

    def run_main_application(self):
        """Run the main air paint application"""
        cv2.namedWindow("Paint", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Output", cv2.WINDOW_AUTOSIZE)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = self.hands.process(frame_rgb)

            landmarks = []
            if result.multi_hand_landmarks:
                for hand_lms in result.multi_hand_landmarks:
                    lm_list = []
                    for lm in hand_lms.landmark:
                        lmx = int(lm.x * self.WINDOW_WIDTH)
                        lmy = int(lm.y * self.WINDOW_HEIGHT)
                        lm_list.append((lmx, lmy))
                    landmarks = lm_list

                    self.mp_draw.draw_landmarks(
                        frame, hand_lms, self.mp_hands.HAND_CONNECTIONS
                    )

                finger_pos, is_drawing, pinch_event = self.detect_gestures(
                    landmarks, frame.shape
                )

                if finger_pos:
                    cursor_color = (0, 255, 0) if is_drawing else (0, 0, 255)
                    cv2.circle(frame, finger_pos, 8, cursor_color, -1)

                    if pinch_event:
                        button_action = self.detect_button_click(finger_pos)
                        if button_action:
                            self.handle_button_action(button_action)

                    if is_drawing and finger_pos[1] > self.BUTTON_HEIGHT + 20:
                        brush_size = self.brush_sizes[self.current_brush_size_index]
                        if self.current_tool == "eraser":
                            draw_color = (255, 255, 255)
                            brush_size *= 3
                            preview_color = (0, 0, 0)
                        else:
                            draw_color = self.colors[self.current_color_index]
                            preview_color = draw_color

                        cv2.circle(
                            frame,
                            finger_pos,
                            brush_size,
                            preview_color,
                            2,
                        )

                        if self.last_pos is not None:
                            cv2.line(
                                self.paint_window,
                                self.last_pos,
                                finger_pos,
                                draw_color,
                                brush_size * 2,
                            )
                        self.last_pos = finger_pos

                if not is_drawing:
                    if self.prev_drawing:
                        self.save_to_history()
                        self.last_pos = None

                self.prev_drawing = is_drawing

            else:
                if self.prev_drawing:
                    self.save_to_history()
                    self.last_pos = None
                self.prev_drawing = False

            # Overlay canvas on frame
            mask = np.any(self.paint_window != 255, axis=-1, keepdims=True)
            frame = np.where(mask, self.paint_window, frame)

            self.draw_ui_elements(frame)

            cv2.imshow("Paint", frame)
            cv2.imshow("Output", self.paint_window)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s") and self.tools["save"]:
                self.save_drawing()
            elif key == ord("z") and self.tools["undo_redo"]:
                self.undo()
            elif key == ord("y") and self.tools["undo_redo"]:
                self.redo()

        self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        """Main entry point"""
        self.run_settings()


if __name__ == "__main__":
    app = AirPaintApp()
    app.run()
