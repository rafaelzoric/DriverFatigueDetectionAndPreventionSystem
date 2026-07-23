import RPi.GPIO as GPIO
import time

SPEAKER_PIN = 18
frequency = 1000

def playSpeaker(frequency, SPEAKER_PIN):
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(SPEAKER_PIN, GPIO.OUT)
    
    pwm = GPIO.PWM(SPEAKER_PIN, frequency)
    pwm.start(50)

    print("Speaker playing")
    try:
        time.sleep(3)
    except KeyboardInterrupt:
        pass

    pwm.stop()
    del pwm
    GPIO.cleanup()
    print("Done")
    
playSpeaker(frequency, SPEAKER_PIN);