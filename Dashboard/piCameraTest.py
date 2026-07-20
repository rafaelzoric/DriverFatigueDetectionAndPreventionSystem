from picamera2 import Picamera2
import cv2

# Start Pi Camera (XC9021 IR camera via CSI port)
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}
)
picam2.configure(config)
picam2.start()

print("Pi Camera connected — press Q to quit")

while True:
    # Read frame
    frame = picam2.capture_array()

    # Show camera feed
    cv2.imshow("Pi Camera Test", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
picam2.stop()
