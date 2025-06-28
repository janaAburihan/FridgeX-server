import RPi.GPIO as GPIO
import threading
import time
import requests
import firebase_admin
from firebase_admin import credentials, messaging

DOOR_SENSOR_PIN = 14
door_status = {"is_open": None}
open_start_time = None
notification_sent = False
gpio_initialized = False

# Initialize Firebase Admin SDK
try:
    cred = credentials.Certificate("/home/JSL/Desktop/FridgeX/fridgex-93c26-firebase-adminsdk-fbsvc-9d9f44769d.json")  
    firebase_admin.initialize_app(cred)
    print("[FIREBASE] Initialized Firebase Admin SDK.")
except Exception as e:
    print(f"[ERROR] Firebase initialization failed: {e}")

def get_fcm_token():
    try:
        with open("fcm_token.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        print("[ERROR] FCM token file not found.")
        return None

def send_notification():
    try:
        token = get_fcm_token()
        if not token:
            print("[NOTIFICATION] No FCM token found. Skipping.")
            return

        message = messaging.Message(
            notification=messaging.Notification(
                title="Fridge Door Alert ??",
                body="The fridge door has been left open for over 2 minutes!",
            ),
            data={
                "click_action": "FLUTTER_NOTIFICATION_CLICK",
                "status": "door_open"
            },
            token=token,
        )

        print("[NOTIFICATION] Sending notification via Firebase Admin SDK...")
        response = messaging.send(message)
        print("[NOTIFICATION] Successfully sent:", response)

    except Exception as e:
        print(f"[ERROR] Failed to send notification: {e}")

def init_gpio():
    global gpio_initialized
    if not gpio_initialized:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(DOOR_SENSOR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        gpio_initialized = True
        print("[GPIO] Initialized")

def monitor_door():
    global open_start_time, notification_sent
    if not gpio_initialized:
        init_gpio()

    while True:
        try:
            state = GPIO.input(DOOR_SENSOR_PIN)
            is_open = (state == GPIO.HIGH)
            door_status["is_open"] = is_open

            if is_open:
                if open_start_time is None:
                    open_start_time = time.time()
                    notification_sent = False
                elif not notification_sent and time.time() - open_start_time > 120:
                    send_notification()
                    notification_sent = True
            else:
                open_start_time = None
                notification_sent = False

            time.sleep(1)
        except RuntimeError as e:
            print(f"[GPIO ERROR] {e}")
            break

def start_door_monitoring():
    init_gpio()
    thread = threading.Thread(target=monitor_door, daemon=True)
    thread.start()

def get_door_status():
    return door_status["is_open"]
