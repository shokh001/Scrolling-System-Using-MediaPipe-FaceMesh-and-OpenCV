import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui
from collections import deque
from flask import Flask, render_template_string, Response
import threading

class WebPupilScroller:
    def __init__(self):
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
        
        # Qorachiq pozitsiyasi
        self.pupil_history = deque(maxlen=5)
        self.last_direction = "MARKAZ"
        self.last_scroll_time = 0
        self.scroll_cooldown = 0.3  # 300ms
        
        # Kalibratsiya
        self.calibration_complete = False
        self.calibration_samples = []
        self.center_x, self.center_y = 0.5, 0.5
        
        # YANGI: Teng chegara koeffitsientlari
        self.base_threshold_y = 0.045  # Asosiy chegara qiymati
        self.threshold_y = self.base_threshold_y
        
        # YANGI: Teng chegara koeffitsientlari
        self.upper_multiplier = 1.0  # Tepa chegara
        self.lower_multiplier = 1.0  # Past chegara
        
        self.scroll_speed = 18
        
        # Debug
        self.frame_count = 0
        self.last_debug_time = 0

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
                print(f"✅ Kalibratsiya tugadi! Markaz: ({self.center_x:.3f}, {self.center_y:.3f})")
                print(f"🔧 Teng chegara: ±{self.base_threshold_y:.3f}")
                print("🎯 Endi qorachiqni harakatlantiring:")
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
        cv2.rectangle(frame, (10, 10), (650, 200), (40, 40, 40), -1)
        cv2.rectangle(frame, (10, 10), (650, 200), (255, 255, 255), 1)
        
        status_color = (0, 255, 0) if self.calibration_complete else (0, 255, 255)
        
        info_text = [
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

    def process_frame(self, frame):
        """Asosiy kadrni qayta ishlash"""
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
                    direction = "XATOLIK"
        
        self.draw_web_interface(frame, direction, pupil_x, pupil_y)
        
        return frame

    def start_webcam(self):
        """Webcam ni ishga tushirish"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("🌐 WEB QORACHIQ SKROLL PLUGIN - TENG CHEGARA")
        print("=" * 70)
        print("🔧 YANGI SOZLAMALAR:")
        print(f"   • Teng chegara: ±{self.base_threshold_y:.3f}")
        print(f"   • Tepa chegara koeffitsienti: {self.upper_multiplier}x")
        print(f"   • Past chegara koeffitsienti: {self.lower_multiplier}x")
        print(f"   • Skroll tezligi: {self.scroll_speed}px")
        print("🎯 AFZALLIKLARI:")
        print("   • Tepa va past chegara TENG")
        print("   • Ikkala tomonga bir xil harakat talab qilinadi")
        print("   • Balanslangan va tabiiy his qilinadi")
        print("⏹️  To'xtatish: 'q' tugmasi")
        print("=" * 70)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = self.process_frame(frame)
                cv2.imshow('🌐 Web Qorachiq Skroll - TENG CHEGARA', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            print(f"Xatolik: {e}")
            
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\n✅ Dastur to'xtatildi!")

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
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 200vh;
        }
        .container {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            margin: 20px 0;
        }
        h1 {
            text-align: center;
            font-size: 2.8em;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            font-size: 1.3em;
            opacity: 0.9;
            margin-bottom: 30px;
        }
        .test-info {
            background: rgba(0,255,255,0.2);
            padding: 20px;
            border-radius: 15px;
            margin: 20px 0;
            border-left: 5px solid #00ffff;
        }
        .instructions {
            background: rgba(255,255,255,0.2);
            padding: 25px;
            border-radius: 15px;
            margin: 25px 0;
        }
        .instruction-item {
            display: flex;
            align-items: center;
            margin: 18px 0;
            font-size: 1.2em;
            padding: 10px;
            border-radius: 10px;
            background: rgba(255,255,255,0.1);
        }
        .icon {
            font-size: 2.5em;
            margin-right: 20px;
            width: 60px;
            text-align: center;
        }
        .status {
            text-align: center;
            font-size: 1.4em;
            margin: 25px 0;
            padding: 20px;
            background: rgba(0,255,0,0.3);
            border-radius: 15px;
        }
        .content-section {
            background: rgba(255,255,255,0.1);
            padding: 30px;
            margin: 25px 0;
            border-radius: 15px;
            line-height: 1.8;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌐 Web Qorachiq Skroll - TENG CHEGARA</h1>
        <div class="subtitle">Tepa va past chegara teng - balanslangan</div>
        
        <div class="test-info">
            <strong>⚖️ TENG CHEGARA SOZLAMASI</strong><br>
            • Tepa chegara: ±{chegara}<br>
            • Past chegara: ±{chegara}<br>
            • Ikkala tomonga bir xil harakat talab qilinadi
        </div>
        
        <div class="instructions">
            <h2>🎯 Test Qadamları:</h2>
            
            <div class="instruction-item">
                <div class="icon">👆</div>
                <div>
                    <strong>Qorachiqni TEPAGA qarating</strong><br>
                    → Tepa chegara: markaz - {chegara}<br>
                    → Terminalda "🖱️ Skroll: PASTGA" xabarini ko'ring
                </div>
            </div>
            
            <div class="instruction-item">
                <div class="icon">👇</div>
                <div>
                    <strong>Qorachiqni PASTGA qarating</strong><br>
                    → Past chegara: markaz + {chegara}<br>
                    → Terminalda "🖱️ Skroll: YUQORIGA" xabarini ko'ring
                </div>
            </div>
            
            <div class="instruction-item">
                <div class="icon">⚖️</div>
                <div>
                    <strong>Teng chegara afzalliklari</strong><br>
                    • Tepa va past chegara bir xil<br>
                    • Ikkala tomonga bir xil harakat<br>
                    • Balanslangan va tabiiy his qilinadi
                </div>
            </div>
        </div>

        <div id="status" class="status">
            🔄 Kamera ishga tushirilmoqda... Terminalni kuzating
        </div>

        <div class="content-section">
            <h3>📖 Test Kontenti (Skroll qilish uchun)</h3>
            <p>Bu uzun matn skroll qilishni sinab ko'rish uchun. Terminalda skroll xabarlarini kuzating.</p>
            
            <!-- Uzun kontent -->
            <h4>Birinchi bo'lim</h4>
            <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.</p>
            
            <h4>Ikkinchi bo'lim</h4>
            <p>Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.</p>
            
            <h4>Uchinchi bo'lim</h4>
            <p>Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.</p>
            
            <h4>To'rtinchi bo'lim</h4>
            <p>Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.</p>
            
            <h4>Beshinchi bo'lim</h4>
            <p>Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium.</p>
            
            <h4>Oltinchi bo'lim</h4>
            <p>Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores.</p>
            
            <h4>Yettinchi bo'lim</h4>
            <p>Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit.</p>
        </div>
    </div>

    <script>
        document.body.style.height = '3000px';
        
        function updateStatus(message) {
            document.getElementById('status').textContent = message;
        }
        
        setTimeout(() => {
            updateStatus("✅ Kamera faol - Teng chegara sozlamalari faollashtirildi!");
        }, 3000);
    </script>
</body>
</html>
    ''')

def run_flask():
    """Flask serverni ishga tushirish"""
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

def main():
    """Asosiy dastur"""
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    scroller = WebPupilScroller()
    
    print("🌐 Web Plugin ishga tushirilmoqda...")
    print("📍 Web interfeys: http://localhost:5000")
    print("🔧 Terminalda teng chegara sozlamalarini kuzating!")
    
    try:
        scroller.start_webcam()
    except KeyboardInterrupt:
        print("\nDastur to'xtatildi.")
    finally:
        print("✅ Web Plugin to'xtatildi")

if __name__ == "__main__":
    main()