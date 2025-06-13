# lock_unlock.py
import RPi.GPIO as GPIO
import time

SERVO_PIN = 22 
GPIO.setmode(GPIO.BCM)

def setup():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(SERVO_PIN, GPIO.OUT)
    pwm = GPIO.PWM(SERVO_PIN, 50)  # 50Hz PWM frequency
    pwm.start(0)
    return pwm

def move_servo(pwm, duty_cycle):
    pwm.ChangeDutyCycle(duty_cycle)
    time.sleep(0.5)
    pwm.ChangeDutyCycle(0)

def lock_door():
    pwm = setup()
    print("?? Locking the door...")
    move_servo(pwm, 2.5)  # 0 degrees (adjust as needed)
    pwm.stop()
    GPIO.cleanup()

def unlock_door():
    pwm = setup()
    print("?? Unlocking the door...")
    move_servo(pwm, 7.5)  # 90 degrees (adjust as needed)
    pwm.stop()
    GPIO.cleanup()
