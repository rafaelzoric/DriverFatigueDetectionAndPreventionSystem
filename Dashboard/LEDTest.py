import RPi.GPIO as GPIO
import time

LED_PIN = 17
frequency = 2500

def LED_PWM(LED_PIN, frequency):
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LED_PIN, GPIO.OUT)
    
    pwm = GPIO.PWM(LED_PIN, frequency)
    pwm.start(50)
    
    print("LED On")
    try:
        time.sleep(3)
    except KeyboardInterrupt:
        pass
    pwm.stop()
    del pwm
    GPIO.cleanup()
    print("Done")
    
LED_PWM(LED_PIN, frequency);
