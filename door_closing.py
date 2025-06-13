import RPi.GPIO as GPIO
import time

SERVO_PIN = 17  # Update to your actual GPIO pin

def setup():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(SERVO_PIN, GPIO.OUT)
    pwm = GPIO.PWM(SERVO_PIN, 50)  # 50Hz for servo
    pwm.start(0)
    return pwm

def close_door():
    pwm = setup()
    pwm.ChangeDutyCycle(12.5)  # rotate to 180 degrees
    time.sleep(0.5)

    pwm.ChangeDutyCycle(0)  # stop sending signal
    pwm.stop()
    # GPIO.cleanup()


