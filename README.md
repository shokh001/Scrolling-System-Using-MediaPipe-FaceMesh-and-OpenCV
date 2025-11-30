# 🌐 Ko‘z Qorachig‘i Harakati Orqali Real-Vaqt Skroll Boshqaruv Tizimi  
### MediaPipe FaceMesh + OpenCV + PyAutoGUI asosida

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Ishlayabdi-brightgreen)

## 🇺🇿 O‘zbekcha Tavsif

Bu loyiha **ko‘z qorachig‘ining vertikal harakati** orqali kompyuterda sahifani yuqoriga yoki pastga skroll qilish imkonini beradi.  

**Asosiy xususiyatlar:**
- Tepa va pastga qarash chegaralari **teng** (balanslangan sezgirlik)
- Avtomatik kalibratsiya (60 kadr ichida markaz aniqlanadi)
- Dinamik threshold – foydalanuvchi qanchalik uzoqqa qarasa, skroll shunchalik tezlashadi
- Chiroyli vizual interfeys (qorachiq pozitsiyasi grafigi, ko‘z konturlari)
- PyAutoGUI bilan ishonchli skroll
- Flask orqali chiroyli test sahifasi[](http://localhost:5000)

> Endi klaviatura yoki sichqoncha kerak emas — faqat ko‘z bilan skroll qiling!

---

## 🇬🇧 English Description

A real-time **eye-controlled scrolling system** using pupil vertical movement detection via **MediaPipe FaceMesh** and **OpenCV**.

**Key Features:**
- **Perfectly balanced sensitivity** – equal threshold for looking up and down
- Automatic calibration (center point calculated in ~2 seconds)
- Adaptive threshold – the further you look, the faster it scrolls
- Beautiful on-screen overlay with pupil position graph and eye contours
- Reliable scrolling using PyAutoGUI
- Built-in Flask web interface for testing[](http://localhost:5000)

> Scroll any webpage hands-free — just with your eyes!

---

## 🚀 Qanday ishlatiladi?

### 1. Kerakli kutubxonalar
```bash
pip install opencv-python mediapipe pyautogui flask numpy
