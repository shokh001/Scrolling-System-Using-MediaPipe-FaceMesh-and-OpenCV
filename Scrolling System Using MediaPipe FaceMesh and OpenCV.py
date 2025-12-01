import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui
from collections import deque
from flask import Flask, render_template_string, Response
import threading
import sys
import platform

class WebPupilScroller:
    def __init__(self, camera_index=0):
        # MediaPipe sozlamalari
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Ko'z va qorachiq nuqtalari
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        self.LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Kamerani tanlash
        self.camera_index = camera_index
        self.camera_list = []
        self.detect_available_cameras()
        
        # Qorachiq pozitsiyasi
        self.pupil_history = deque(maxlen=5)
        self.last_direction = "MARKAZ"
        self.last_scroll_time = 0
        self.scroll_cooldown = 0.3  # 300ms
        
        # Kalibratsiya
        self.calibration_complete = False
        self.calibration_samples = []
        self.center_x, self.center_y = 0.5, 0.5
        
        # Teng chegara koeffitsientlari
        self.base_threshold_y = 0.045  # Asosiy chegara qiymati
        self.threshold_y = self.base_threshold_y
        
        # Teng chegara koeffitsientlari
        self.upper_multiplier = 1.0  # Tepa chegara
        self.lower_multiplier = 1.0  # Past chegara
        
        self.scroll_speed = 18
        
        # Debug
        self.frame_count = 0
        self.last_debug_time = 0
        self.cap = None
        
        # Kamera sozlamalari
        self.camera_width = 1280
        self.camera_height = 720
        self.fps = 30

    def detect_available_cameras(self):
        """Mavjud kameralarni aniqlash"""
        print("🔍 Mavjud kameralarni aniqlash...")
        available_cameras = []
        
        # Turli platformalar uchun kamera test
        if platform.system() == 'Windows':
            test_range = 10
        elif platform.system() == 'Linux':
            test_range = 10
        else:  # macOS
            test_range = 10
            
        for i in range(test_range):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if platform.system() == 'Windows' else cv2.CAP_ANY)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    available_cameras.append(i)
                    print(f"   ✅ Kamera {i} topildi - {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
                cap.release()
            else:
                print(f"   ❌ Kamera {i} mavjud emas")
                
        self.camera_list = available_cameras
        if not available_cameras:
            print("⚠️ Hech qanday kamera topilmadi!")
            print("Qo'shimcha kamera ulangandan keyin qayta urinib ko'ring.")
            sys.exit(1)
            
        return available_cameras

    def select_camera(self):
        """Kamerani tanlash"""
        if not self.camera_list:
            print("❌ Mavjud kamera yo'q!")
            return False
            
        print(f"\n📹 Mavjud kameralar: {self.camera_list}")
        
        if self.camera_index >= len(self.camera_list):
            print(f"⚠️ {self.camera_index}-kamera mavjud emas, 0-kameradan foydalanilmoqda")
            self.camera_index = 0
            
        # Kamerani ochish
        print(f"\n🔄 {self.camera_index}-kamera ochilmoqda...")
        
        # Platformaga qarab backend tanlash
        if platform.system() == 'Windows':
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.camera_index)
            
        if not self.cap.isOpened():
            print(f"❌ {self.camera_index}-kamera ochilmadi!")
            
            # Boshqa kamerani sinab ko'rish
            for cam_idx in self.camera_list:
                if cam_idx != self.camera_index:
                    print(f"🔄 {cam_idx}-kamerani sinab ko'ramiz...")
                    self.cap = cv2.VideoCapture(cam_idx)
                    if self.cap.isOpened():
                        self.camera_index = cam_idx
                        print(f"✅ {cam_idx}-kamera muvaffaqiyatli ochildi!")
                        break
                        
        if not self.cap.isOpened():
            print("❌ Hech qanday kamera ochilmadi!")
            return False
            
        # Kamera parametrlarini sozlash
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Sozlamalarni tekshirish
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✅ Kamera {self.camera_index} muvaffaqiyatli ochildi!")
        print(f"📐 O'lcham: {actual_width}x{actual_height}")
        print(f"⚡ FPS: {actual_fps}")
        
        return True

    def get_pupil_relative_position(self, landmarks):
        """Qorachiqning nisbiy pozitsiyasini aniq hisoblash"""
        try:
            # Chap ko'z qorachig'i
            left_iris_points = np.array([(landmarks.landmark[idx].x, landmarks.landmark[idx].y) for idx in self.LEFT_IRIS])
            left_center = np.mean(left_iris_points, axis=0)
            
            # Chap ko'z chegara
            left_eye_points = np.array([(landmarks.landmark[idx].x, landmarks.landmark[idx].y) for idx in self.LEFT_EYE])
            min_x, min_y = np.min(left_eye_points, axis=0)
            max_x, max_y = np.max(left_eye_points, axis=0)
            left_relative_x = (left_center[0] - min_x) / (max_x - min_x)
            left_relative_y = (left_center[1] - min_y) / (max_y - min_y)
            
            # O'ng ko'z qorachig'i
            right_iris_points = np.array([(landmarks.landmark[idx].x, landmarks.landmark[idx].y) for idx in self.RIGHT_IRIS])
            right_center = np.mean(right_iris_points, axis=0)
            
            # O'ng ko'z chegara
            right_eye_points = np.array([(landmarks.landmark[idx].x, landmarks.landmark[idx].y) for idx in self.RIGHT_EYE])
            min_x, min_y = np.min(right_eye_points, axis=0)
            max_x, max_y = np.max(right_eye_points, axis=0)
            right_relative_x = (right_center[0] - min_x) / (max_x - min_x)
            right_relative_y = (right_center[1] - min_y) / (max_y - min_y)
            
            # O'rtacha
            avg_x = (left_relative_x + right_relative_x) / 2
            avg_y = (left_relative_y + right_relative_y) / 2
            
            return max(0.0, min(1.0, avg_x)), max(0.0, min(1.0, avg_y))
            
        except Exception as e:
            print(f"⚠️ Qorachiq pozitsiyasini hisoblashda xatolik: {e}")
            return 0.5, 0.5

    def calibrate(self, pupil_x, pupil_y):
        """Kalibratsiya - YANGILANGAN"""
        if not self.calibration_complete:
            self.calibration_samples.append((pupil_x, pupil_y))
            
            if len(self.calibration_samples) >= 60:
                xs = [p[0] for p in self.calibration_samples]
                ys = [p[1] for p in self.calibration_samples]
                
                self.center_x = np.median(xs)
                self.center_y = np.median(ys)
                
                # YANGI: Dinamik threshold - teng chegara
                std_y = np.std(ys)
                self.base_threshold_y = max(0.04, 1.3 * std_y)
                self.threshold_y = self.base_threshold_y
                
                self.calibration_complete = True
                print(f"\n✅ Kalibratsiya tugadi!")
                print(f"📍 Markaz: ({self.center_x:.3f}, {self.center_y:.3f})")
                print(f"⚖️ Teng chegara: ±{self.base_threshold_y:.3f}")
                print("\n🎯 Endi qorachiqni harakatlantiring:")
                print(f"   👆 TEPAGA (Y < {self.center_y - self.base_threshold_y:.3f}) → PASTGA skroll")
                print(f"   👇 PASTGA (Y > {self.center_y + self.base_threshold_y:.3f}) → YUQORIGA skroll")
                print("💡 TENG CHEGARA - IKKALA TOMON BIR XIL!")
            
            return True
        return False

    def detect_and_scroll(self, pupil_x, pupil_y):
        """Yo'nalishni aniqlash va skroll qilish - YANGILANGAN"""
        current_time = time.time()
        
        if self.calibrate(pupil_x, pupil_y):
            return "KALIBRATSIYA...", pupil_x, pupil_y
        
        # Smooth qilish
        self.pupil_history.append((pupil_x, pupil_y))
        smooth_x = np.mean([p[0] for p in self.pupil_history])
        smooth_y = np.mean([p[1] for p in self.pupil_history])
        
        # YANGI: Teng chegara hisoblash
        upper_threshold = self.base_threshold_y * self.upper_multiplier
        lower_threshold = self.base_threshold_y * self.lower_multiplier
        
        # Progress hisoblash
        progress_upper = 0
        progress_lower = 0
        
        if smooth_y < (self.center_y - upper_threshold):
            progress_upper = (self.center_y - smooth_y) / upper_threshold
        elif smooth_y > (self.center_y + lower_threshold):
            progress_lower = (smooth_y - self.center_y) / lower_threshold
        
        # Progress asosida chegara moslashuvi
        adaptive_upper = upper_threshold * max(0.7, min(1.5, 1.0 + progress_upper * 0.2))
        adaptive_lower = lower_threshold * max(0.7, min(1.5, 1.0 + progress_lower * 0.2))
        
        # Yo'nalishni aniqlash - YANGI chegara bilan
        direction = "MARKAZ"
        scroll_amount = 0
        
        # Tepa yoki past - YANGI chegara bilan
        if smooth_y < (self.center_y - adaptive_upper):
            direction = "TEPAGA 👆"
            scroll_amount = self.scroll_speed  # Pastga skroll
        elif smooth_y > (self.center_y + adaptive_lower):
            direction = "PASTGA 👇" 
            scroll_amount = -self.scroll_speed  # Yuqoriga skroll
        
        # Skroll qilish
        if scroll_amount != 0 and current_time - self.last_scroll_time > self.scroll_cooldown:
            try:
                # Progressga qarab skroll miqdorini o'zgartirish
                progress = max(progress_upper, progress_lower)
                adjusted_scroll = int(scroll_amount * min(2.5, 1.0 + progress))
                
                pyautogui.scroll(adjusted_scroll)
                self.last_scroll_time = current_time
                
                print(f"🖱️ Skroll: {'PASTGA' if scroll_amount > 0 else 'YUQORIGA'} ({abs(adjusted_scroll)} px) | Progress: {progress:.1f}x")
                
            except Exception as e:
                print(f"❌ Skroll xatosi: {e}")
                # Alternative skroll usuli
                try:
                    if scroll_amount > 0:
                        pyautogui.press('down', presses=2)
                        print("🔧 Alternative: DOWN tugmasi ishlatildi")
                    else:
                        pyautogui.press('up', presses=2)
                        print("🔧 Alternative: UP tugmasi ishlatildi")
                except Exception as e2:
                    print(f"❌ Alternative skroll ham ishlamadi: {e2}")
        
        # Debug - har 1.5 sekundda
        if current_time - self.last_debug_time > 1.5:
            print(f"📊 Y:{smooth_y:.3f} | Tepa chegara: {self.center_y - adaptive_upper:.3f} | Past chegara: {self.center_y + adaptive_lower:.3f} | {direction}")
            self.last_debug_time = current_time
        
        # Alert faqat yo'nalish o'zgarganda
        if direction != self.last_direction:
            print(f"👁️ Yo'nalish: {direction}")
            self.last_direction = direction
        
        return direction, smooth_x, smooth_y

    def draw_web_interface(self, frame, direction, pupil_x, pupil_y):
        """Web interfeysini chizish - YANGILANGAN"""
        h, w = frame.shape[:2]
        
        # Asosiy ma'lumotlar paneli
        cv2.rectangle(frame, (10, 10), (700, 200), (40, 40, 40), -1)
        cv2.rectangle(frame, (10, 10), (700, 200), (255, 255, 255), 1)
        
        status_color = (0, 255, 0) if self.calibration_complete else (0, 255, 255)
        
        info_text = [
            f"Kamera: {self.camera_index} | O'lcham: {w}x{h}",
            f"Qorachiq Y: {pupil_y:.3f} | Markaz: {self.center_y:.3f}",
            f"Yo'nalish: {direction}",
            f"Holat: {'✅ TENG CHEGARA SKROLL' if self.calibration_complete else '🔄 KALIBRATSIYA'}",
            f"Teng chegara: ±{self.base_threshold_y:.3f}",
            f"Tepa chegara: {self.center_y - self.base_threshold_y:.3f} | Past chegara: {self.center_y + self.base_threshold_y:.3f}",
            f"Skroll tezligi: {self.scroll_speed}px | Cooldown: {self.scroll_cooldown*1000}ms",
            f"Sezgirlik: BALANS | Tepa va past teng"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (20, 35 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # Katta yo'nalish ko'rsatkichi
        center_x, center_y = w // 2, h // 2
        
        # Ranglar
        color_map = {
            "TEPAGA 👆": (0, 255, 0),    # Yashil
            "PASTGA 👇": (0, 0, 255),    # Qizil
            "MARKAZ": (255, 255, 255),   # Oq
            "KALIBRATSIYA...": (0, 255, 255) # Sariq
        }
        
        color = color_map.get(direction, (255, 255, 255))
        
        # Katta emoji va matn
        if "TEPAGA" in direction:
            cv2.putText(frame, "👆", (center_x, center_y - 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 4, color, 8)
            cv2.putText(frame, "TEPAGA → PASTGA SKROLL", (center_x - 200, center_y - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, f"Skroll miqdori: {self.scroll_speed}px", (center_x - 100, center_y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        elif "PASTGA" in direction:
            cv2.putText(frame, "👇", (center_x, center_y + 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 4, color, 8)
            cv2.putText(frame, "PASTGA → YUQORIGA SKROLL", (center_x - 220, center_y + 250), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, f"Skroll miqdori: {self.scroll_speed}px", (center_x - 100, center_y + 290), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            cv2.putText(frame, "◎", (center_x, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 4, color, 8)
            cv2.putText(frame, "MARKAZ - SKROLL YO'Q", (center_x - 180, center_y + 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Kamera ma'lumoti
        cv2.putText(frame, f"📹 Kamera #{self.camera_index}", (w - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Qorachiq pozitsiyasi grafigi
        self.draw_position_graph(frame, pupil_x, pupil_y)

    def draw_position_graph(self, frame, x, y):
        """Qorachiq pozitsiyasi grafigi - YANGILANGAN"""
        h, w = frame.shape[:2]
        size = 200
        graph_x, graph_y = w - size - 20, 20
        
        # Grafik asosi
        cv2.rectangle(frame, (graph_x, graph_y), 
                     (graph_x + size, graph_y + size), (30, 30, 30), -1)
        cv2.rectangle(frame, (graph_x, graph_y), 
                     (graph_x + size, graph_y + size), (200, 200, 200), 2)
        
        # Markaz chizig'lari
        center_x = int(self.center_x * size)
        center_y = int(self.center_y * size)
        cv2.line(frame, (graph_x + center_x, graph_y), 
                (graph_x + center_x, graph_y + size), (100, 100, 100), 1)
        cv2.line(frame, (graph_x, graph_y + center_y), 
                (graph_x + size, graph_y + center_y), (100, 100, 100), 1)
        
        # YANGI: Teng chegara chizig'lari
        upper_y = int((self.center_y - self.base_threshold_y) * size)
        lower_y = int((self.center_y + self.base_threshold_y) * size)
        
        cv2.line(frame, (graph_x, graph_y + max(0, upper_y)), 
                (graph_x + size, graph_y + max(0, upper_y)), (100, 255, 100), 3)
        cv2.line(frame, (graph_x, graph_y + min(size, lower_y)), 
                (graph_x + size, graph_y + min(size, lower_y)), (100, 100, 255), 3)
        
        # Progress chizig'i
        current_y = graph_y + int(y * size)
        if y < self.center_y - self.base_threshold_y:
            # Tepaga progress
            cv2.rectangle(frame, (graph_x, current_y), 
                         (graph_x + size, center_y - upper_y), (0, 255, 0, 100), -1)
        elif y > self.center_y + self.base_threshold_y:
            # Pastga progress
            cv2.rectangle(frame, (graph_x, center_y + lower_y), 
                         (graph_x + size, current_y), (0, 0, 255, 100), -1)
        
        # Qorachiq nuqtasi
        dot_x = graph_x + int(x * size)
        dot_y = graph_y + int(y * size)
        cv2.circle(frame, (dot_x, dot_y), 6, (0, 0, 255), -1)
        cv2.circle(frame, (dot_x, dot_y), 8, (255, 255, 255), 2)
        
        # Markaz nuqtasi
        if self.calibration_complete:
            center_dot_x = graph_x + center_x
            center_dot_y = graph_y + center_y
            cv2.circle(frame, (center_dot_x, center_dot_y), 4, (0, 255, 0), -1)
        
        # Belgilar
        cv2.putText(frame, f"TEPA (<{self.center_y - self.base_threshold_y:.2f})", 
                   (graph_x + 5, graph_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"PAST (>{self.center_y + self.base_threshold_y:.2f})", 
                   (graph_x + 5, graph_y + size - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "TENG CHEGARA", (graph_x + 5, graph_y + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    def draw_eye_contours(self, frame, landmarks):
        """Ko'z chegaralarini yashil chiziq bilan chizish"""
        try:
            h, w = frame.shape[:2]
            
            # Chap ko'z
            left_points = []
            for idx in self.LEFT_EYE:
                point = landmarks.landmark[idx]
                x = int(point.x * w)
                y = int(point.y * h)
                left_points.append([x, y])
            
            left_points = np.array(left_points, dtype=np.int32)
            cv2.polylines(frame, [left_points], True, (0, 255, 0), 2)
            
            # O'ng ko'z
            right_points = []
            for idx in self.RIGHT_EYE:
                point = landmarks.landmark[idx]
                x = int(point.x * w)
                y = int(point.y * h)
                right_points.append([x, y])
            
            right_points = np.array(right_points, dtype=np.int32)
            cv2.polylines(frame, [right_points], True, (0, 255, 0), 2)
            
            # Qorachig'larni chizish
            left_iris_points = []
            for idx in self.LEFT_IRIS:
                point = landmarks.landmark[idx]
                x = int(point.x * w)
                y = int(point.y * h)
                left_iris_points.append([x, y])
            
            left_center = np.mean(left_iris_points, axis=0).astype(int)
            cv2.circle(frame, tuple(left_center), 4, (0, 0, 255), -1)
            
            right_iris_points = []
            for idx in self.RIGHT_IRIS:
                point = landmarks.landmark[idx]
                x = int(point.x * w)
                y = int(point.y * h)
                right_iris_points.append([x, y])
            
            right_center = np.mean(right_iris_points, axis=0).astype(int)
            cv2.circle(frame, tuple(right_center), 4, (0, 0, 255), -1)
            
        except Exception as e:
            print(f"⚠️ Ko'z chizishda xatolik: {e}")

    def process_frame(self, frame):
        """Asosiy kadrni qayta ishlash"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            direction = "YUZ ANIQLANMADI"
            pupil_x, pupil_y = 0.5, 0.5
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    try:
                        pupil_x, pupil_y = self.get_pupil_relative_position(face_landmarks)
                        
                        direction, pupil_x, pupil_y = self.detect_and_scroll(pupil_x, pupil_y)
                        
                        self.draw_eye_contours(frame, face_landmarks)
                        break
                        
                    except Exception as e:
                        direction = f"XATOLIK: {str(e)[:20]}"
                        print(f"⚠️ Kadr qayta ishlash xatosi: {e}")
            
            self.draw_web_interface(frame, direction, pupil_x, pupil_y)
            
            return frame
            
        except Exception as e:
            print(f"❌ Kadr qayta ishlashda jiddiy xatolik: {e}")
            # Xatolik holatida oddiy matn chizish
            cv2.putText(frame, "XATOLIK: Kadr qayta ishlash", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame

    def start_webcam(self):
        """Webcam ni ishga tushirish - YANGILANGAN"""
        # Kamerani tanlash
        if not self.select_camera():
            return
            
        print("\n" + "="*80)
        print("🌐 WEB QORACHIQ SKROLL PLUGIN - TENG CHEGARA")
        print("="*80)
        print("\n📊 MA'LUMOTLAR:")
        print(f"   • Platforma: {platform.system()} {platform.release()}")
        print(f"   • Kamera: #{self.camera_index}")
        print(f"   • O'lcham: {self.camera_width}x{self.camera_height}")
        print(f"   • Mavjud kameralar: {self.camera_list}")
        
        print("\n🔧 YANGI SOZLAMALAR:")
        print(f"   • Teng chegara: ±{self.base_threshold_y:.3f}")
        print(f"   • Tepa chegara koeffitsienti: {self.upper_multiplier}x")
        print(f"   • Past chegara koeffitsienti: {self.lower_multiplier}x")
        print(f"   • Skroll tezligi: {self.scroll_speed}px")
        
        print("\n🎯 AFZALLIKLARI:")
        print("   • Tepa va past chegara TENG")
        print("   • Ikkala tomonga bir xil harakat talab qilinadi")
        print("   • Balanslangan va tabiiy his qilinadi")
        
        print("\n🎮 QO'LLANMA:")
        print("   1. Kamera oldida to'g'ri o'tiring")
        print("   2. Kalibratsiya tugaguncha (3-5 soniya) to'g'ri qarang")
        print("   3. Qorachiqni TEPAGA qarating → PASTGA skroll")
        print("   4. Qorachiqni PASTGA qarating → YUQORIGA skroll")
        
        print("\n⏹️  BOSHQARUV:")
        print("   • To'xtatish: 'q' tugmasi")
        print("   • Yangi kamera: 'c' tugmasi")
        print("   • Kalibratsiyani qayta boshlash: 'r' tugmasi")
        print("="*80)
        print("\n🔄 Kalibratsiya boshlanmoqda... (3-5 soniya)")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Kadr o'qishda xatolik!")
                    # Kamerani qayta ochishga urinib ko'rish
                    self.cap.release()
                    if not self.select_camera():
                        break
                    continue
                
                # Kadrni qayta ishlash
                frame = self.process_frame(frame)
                
                # Oynani ko'rsatish
                window_name = f'🌐 Web Qorachiq Skroll - Kamera #{self.camera_index}'
                cv2.imshow(window_name, frame)
                
                # Klaviatura boshqaruvi
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n⏹️  Dastur to'xtatildi foydalanuvchi tomonidan")
                    break
                elif key == ord('c'):
                    # Kamera almashtirish
                    print("\n🔄 Kamera almashtirilmoqda...")
                    self.cap.release()
                    cv2.destroyAllWindows()
                    
                    # Keyingi kamera
                    current_idx = self.camera_list.index(self.camera_index)
                    next_idx = (current_idx + 1) % len(self.camera_list)
                    self.camera_index = self.camera_list[next_idx]
                    
                    if not self.select_camera():
                        break
                        
                elif key == ord('r'):
                    # Kalibratsiyani qayta boshlash
                    print("\n🔄 Kalibratsiya qayta boshlanmoqda...")
                    self.calibration_complete = False
                    self.calibration_samples = []
                    
                elif key == ord('+') or key == ord('='):
                    # Skroll tezligini oshirish
                    self.scroll_speed = min(50, self.scroll_speed + 2)
                    print(f"📈 Skroll tezligi: {self.scroll_speed}px")
                    
                elif key == ord('-'):
                    # Skroll tezligini kamaytirish
                    self.scroll_speed = max(5, self.scroll_speed - 2)
                    print(f"📉 Skroll tezligi: {self.scroll_speed}px")
                    
        except KeyboardInterrupt:
            print("\n\n⚠️  Dastur to'xtatildi (Ctrl+C)")
        except Exception as e:
            print(f"\n❌ Kutilmagan xatolik: {e}")
            
        finally:
            # Manbalarni tozalash
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("\n✅ Dastur to'liq to'xtatildi!")
            print("🎯 Web interfeys hali ham ishlayapti: http://localhost:5000")

# Flask Web Interface
app = Flask(__name__)

@app.route('/')
def home():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>🌐 Web Qorachiq Skroll Plugin - TENG CHEGARA</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #ffffff;
            min-height: 100vh;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            padding: 40px 20px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            margin-bottom: 30px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .header h1 {
            font-size: 3.2em;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #00ff88, #00ccff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 30px rgba(0, 255, 136, 0.3);
        }
        
        .subtitle {
            font-size: 1.4em;
            opacity: 0.9;
            margin-bottom: 20px;
        }
        
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 25px;
            margin: 30px 0;
        }
        
        .feature-card {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            background: rgba(255, 255, 255, 0.12);
        }
        
        .feature-icon {
            font-size: 3em;
            margin-bottom: 15px;
        }
        
        .feature-card h3 {
            font-size: 1.4em;
            margin-bottom: 10px;
            color: #00ff88;
        }
        
        .status-panel {
            background: rgba(0, 40, 85, 0.7);
            border-radius: 15px;
            padding: 30px;
            margin: 30px 0;
            text-align: center;
            border: 2px solid #00ccff;
            box-shadow: 0 0 20px rgba(0, 204, 255, 0.2);
        }
        
        .status-title {
            font-size: 1.8em;
            margin-bottom: 20px;
            color: #00ccff;
        }
        
        .status-message {
            font-size: 1.3em;
            padding: 20px;
            background: rgba(0, 255, 136, 0.1);
            border-radius: 10px;
            border: 1px solid #00ff88;
        }
        
        .instructions {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 30px;
            margin: 30px 0;
        }
        
        .instruction-item {
            display: flex;
            align-items: center;
            padding: 20px;
            margin: 15px 0;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            border-left: 4px solid #00ff88;
        }
        
        .instruction-icon {
            font-size: 2.5em;
            margin-right: 20px;
            min-width: 60px;
            text-align: center;
        }
        
        .content-section {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 35px;
            margin: 30px 0;
        }
        
        .section-title {
            font-size: 1.8em;
            margin-bottom: 20px;
            color: #00ccff;
            border-bottom: 2px solid rgba(0, 204, 255, 0.3);
            padding-bottom: 10px;
        }
        
        .camera-info {
            background: rgba(255, 215, 0, 0.1);
            border: 1px solid gold;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
        }
        
        .control-panel {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        
        .control-btn {
            padding: 12px 25px;
            background: linear-gradient(45deg, #00ccff, #0088ff);
            border: none;
            border-radius: 8px;
            color: white;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .control-btn:hover {
            transform: scale(1.05);
            box-shadow: 0 0 15px rgba(0, 204, 255, 0.4);
        }
        
        .key-shortcut {
            display: inline-block;
            padding: 5px 10px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 5px;
            margin: 0 5px;
            font-family: monospace;
        }
        
        @media (max-width: 768px) {
            .header h1 {
                font-size: 2.5em;
            }
            
            .features-grid {
                grid-template-columns: 1fr;
            }
            
            .instruction-item {
                flex-direction: column;
                text-align: center;
            }
            
            .instruction-icon {
                margin-right: 0;
                margin-bottom: 15px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌐 Web Qorachiq Skroll Plugin</h1>
            <div class="subtitle">Tepa va past chegara teng - balanslangan skroll tizimi</div>
            <div class="camera-info">
                <strong>📹 Qo'shimcha kamera qo'llab-quvvatlangan!</strong><br>
                • Har qanday tashqi kamera<br>
                • Bir nechta kamera almashish<br>
                • Avtomatik kamera aniqlash
            </div>
        </div>
        
        <div class="features-grid">
            <div class="feature-card">
                <div class="feature-icon">⚖️</div>
                <h3>TENG CHEGARA</h3>
                <p>Tepa va past chegara bir xil o'lchamda. Ikkala tomonga bir xil harakat talab qilinadi.</p>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <h3>ANIQ SKROLL</h3>
                <p>Qorachiq harakatiga qarab avtomatik skroll. Progressiv tezlik oshishi bilan.</p>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">🔧</div>
                <h3>MOSLASHUVCHAN</h3>
                <p>Har qanday kamera bilan ishlaydi. Real vaqtda sozlamalarni o'zgartirish mumkin.</p>
            </div>
        </div>
        
        <div class="status-panel">
            <div class="status-title">🎮 BOSHQARUV PANELI</div>
            <div class="control-panel">
                <button class="control-btn" onclick="alert('Terminalda \"c\" tugmasini bosing')">📹 Kamera Almash</button>
                <button class="control-btn" onclick="alert('Terminalda \"r\" tugmasini bosing')">🔄 Kalibratsiya</button>
                <button class="control-btn" onclick="alert('Terminalda \"+\" tugmasini bosing')">📈 Tezlik +</button>
                <button class="control-btn" onclick="alert('Terminalda \"-\" tugmasini bosing')">📉 Tezlik -</button>
            </div>
            <div class="status-message">
                ✅ Kamera faol - Terminalda kalibratsiya va skroll xabarlarini kuzating!
            </div>
        </div>
        
        <div class="instructions">
            <h2 class="section-title">🎯 QO'LLANMA</h2>
            
            <div class="instruction-item">
                <div class="instruction-icon">1️⃣</div>
                <div>
                    <strong>Kamerani sozlang</strong><br>
                    Kamera oldida to'g'ri o'tiring. Yuzingiz aniq ko'rinsin.<br>
                    <span class="key-shortcut">c</span> - Kamera almashish
                </div>
            </div>
            
            <div class="instruction-item">
                <div class="instruction-icon">2️⃣</div>
                <div>
                    <strong>Kalibratsiya</strong><br>
                    Dastur ishga tushganda 3-5 soniya davomida to'g'ri qarang.<br>
                    <span class="key-shortcut">r</span> - Kalibratsiyani qayta boshlash
                </div>
            </div>
            
            <div class="instruction-item">
                <div class="instruction-icon">3️⃣</div>
                <div>
                    <strong>Skroll qilish</strong><br>
                    Qorachiqni TEPAGA qarating → PASTGA skroll<br>
                    Qorachiqni PASTGA qarating → YUQORIGA skroll<br>
                    <span class="key-shortcut">+</span>/<span class="key-shortcut">-</span> - Skroll tezligi
                </div>
            </div>
            
            <div class="instruction-item">
                <div class="instruction-icon">4️⃣</div>
                <div>
                    <strong>To'xtatish</strong><br>
                    Dasturni to'xtatish uchun kamera oynasida:<br>
                    <span class="key-shortcut">q</span> - Chiqish
                </div>
            </div>
        </div>
        
        <div class="content-section">
            <h2 class="section-title">📖 Test Kontenti (Skroll qilish uchun)</h2>
            
            <p>Bu uzun matn skroll qilishni sinab ko'rish uchun. Terminalda skroll xabarlarini kuzating.</p>
            
            <!-- Uzun kontent -->
            <h3>Birinchi bo'lim</h3>
            <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.</p>
            
            <h3>Ikkinchi bo'lim</h3>
            <p>Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.</p>
            
            <h3>Uchinchi bo'lim</h3>
            <p>Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores.</p>
            
            <h3>To'rtinchi bo'lim</h3>
            <p>Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit. Ut enim ad minima veniam, quis nostrum exercitationem ullam corporis suscipit laboriosam.</p>
            
            <h3>Beshinchi bo'lim</h3>
            <p>At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident.</p>
            
            <h3>Oltinchi bo'lim</h3>
            <p>Similique sunt in culpa qui officia deserunt mollitia animi, id est laborum et dolorum fuga. Et harum quidem rerum facilis est et expedita distinctio. Nam libero tempore, cum soluta nobis est eligendi optio.</p>
            
            <h3>Yettinchi bo'lim</h3>
            <p>Cumque nihil impedit quo minus id quod maxime placeat facere possimus, omnis voluptas assumenda est, omnis dolor repellendus. Temporibus autem quibusdam et aut officiis debitis aut rerum necessitatibus.</p>
            
            <h3>Sakkizinchi bo'lim</h3>
            <p>Itaque earum rerum hic tenetur a sapiente delectus, ut aut reiciendis voluptatibus maiores alias consequatur aut perferendis doloribus asperiores repellat. Sed ut perspiciatis unde omnis iste natus error.</p>
        </div>
        
        <div class="status-panel">
            <div class="status-title">📊 Tizim Ma'lumotlari</div>
            <div class="status-message">
                <strong>Platforma:</strong> {{ platform }}<br>
                <strong>Kamera:</strong> Avtomatik aniqlash<br>
                <strong>Chegara:</strong> Teng (±0.045)<br>
                <strong>Holat:</strong> Faol - Terminalni kuzating
            </div>
        </div>
    </div>

    <script>
        // Dinamik yozuvlar
        const statusMessages = [
            "✅ Kamera faol - Terminalda kalibratsiya va skroll xabarlarini kuzating!",
            "🎯 Kalibratsiya tugagach, qorachiqni harakatlantiring",
            "⚡ Real vaqtda skroll - hech qanday tugma bosish shart emas!",
            "📹 Qo'shimcha kamera ulangan bo'lsa, 'c' tugmasi bilan almashing"
        ];
        
        let currentMessage = 0;
        const statusElement = document.querySelector('.status-message');
        
        function rotateStatusMessage() {
            currentMessage = (currentMessage + 1) % statusMessages.length;
            statusElement.innerHTML = statusMessages[currentMessage];
            statusElement.style.animation = 'none';
            setTimeout(() => {
                statusElement.style.animation = 'fadeIn 0.5s ease';
            }, 10);
        }
        
        // Har 8 sekundda xabarni almashtirish
        setInterval(rotateStatusMessage, 8000);
        
        // Scroll progress
        window.addEventListener('scroll', function() {
            const scrollTop = window.pageYOffset;
            const docHeight = document.body.offsetHeight;
            const winHeight = window.innerHeight;
            const scrollPercent = scrollTop / (docHeight - winHeight);
            
            // Background gradient o'zgartirish
            document.body.style.background = `linear-gradient(135deg, 
                rgba(26, 26, 46, ${1 - scrollPercent * 0.5}) 0%, 
                rgba(22, 33, 62, ${1 - scrollPercent * 0.3}) 100%)`;
        });
        
        // Klaviatura qisqartmalari
        document.addEventListener('keydown', function(e) {
            switch(e.key.toLowerCase()) {
                case 'c':
                    alert('Terminalda "c" tugmasini bosing - Kamera almashish');
                    break;
                case 'r':
                    alert('Terminalda "r" tugmasini bosing - Kalibratsiya qayta boshlash');
                    break;
                case '+':
                case '=':
                    alert('Terminalda "+" tugmasini bosing - Skroll tezligini oshirish');
                    break;
                case '-':
                    alert('Terminalda "-" tugmasini bosing - Skroll tezligini kamaytirish');
                    break;
            }
        });
    </script>
</body>
</html>
    ''')

def run_flask():
    """Flask serverni ishga tushirish"""
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)

def main():
    """Asosiy dastur"""
    import argparse
    
    # Command line argumentlarni qo'shish
    parser = argparse.ArgumentParser(description='Web Qorachiq Skroll Plugin - TENG CHEGARA')
    parser.add_argument('--camera', type=int, default=0,
                       help='Kamera indeksi (0, 1, 2, ...) [default: 0]')
    parser.add_argument('--width', type=int, default=1280,
                       help='Kamera kengligi [default: 1280]')
    parser.add_argument('--height', type=int, default=720,
                       help='Kamera balandligi [default: 720]')
    parser.add_argument('--speed', type=int, default=18,
                       help='Skroll tezligi [default: 18]')
    
    args = parser.parse_args()
    
    # Flask serverni background threadda ishga tushirish
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    # WebPupilScroller obyektini yaratish
    scroller = WebPupilScroller(camera_index=args.camera)
    scroller.camera_width = args.width
    scroller.camera_height = args.height
    scroller.scroll_speed = args.speed
    
    print("\n" + "="*80)
    print("🚀 WEB QORACHIQ SKROLL PLUGIN ISHGA TUSHIRILMOQDA...")
    print("="*80)
    print(f"\n📍 Web interfeys: http://localhost:5000")
    print(f"📹 Kamera indeksi: {args.camera}")
    print(f"📐 O'lcham: {args.width}x{args.height}")
    print(f"⚡ Skroll tezligi: {args.speed}px")
    print("\n🔧 Terminalda teng chegara sozlamalarini kuzating!")
    print("ℹ️  Yordam uchun: python script.py --help")
    print("="*80)
    
    try:
        scroller.start_webcam()
    except KeyboardInterrupt:
        print("\n\n⚠️  Dastur to'xtatildi (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Dasturda xatolik: {e}")
    finally:
        print("\n✅ Dastur to'liq to'xtatildi!")
        print("🎯 Web interfeys hali ham ishlayapti: http://localhost:5000")

if __name__ == "__main__":
    main()