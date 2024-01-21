import cv2
import time
from ultralytics import YOLO


class Ball:
    def __init__(self, ball_type):
        self.type = ball_type
        self.prev_avg_y = None
        self.bouncing = False  # To track if the ball is bouncing
        self.bounces = 0
        self.coordinates = []
        self.timestamps = []


def initialize_yolo_model():
    return YOLO('best.pt')


def initialize_camera():
    cap = cv2.VideoCapture(0)
    cap.set(3, 800)
    cap.set(4, 400)
    return cap


def detect_balls(frame, model, balls):
    results = model.track(frame, persist=True)

    for ball in balls:
        for result in results:
            if ball.type in result.names.values():
                ball_indices = [i for i, name in result.names.items() if name == ball.type]

                for idx in ball_indices:
                    if len(result.boxes) > idx:
                        bbox = result.boxes[idx]
                        x1, y1, x2, y2 = bbox.xyxy[0].tolist()

                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)

                        ball.coordinates.append((center_x, center_y))
                        ball.timestamps.append(time.time())

                        cv2.putText(frame, f'{ball.type.capitalize()} ({center_x}, {center_y})',
                                    (center_x + 10, center_y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


def calculate_bounces(frame, balls):
    for ball in balls:
        coordinates = ball.coordinates
        if coordinates:
            avg_x = sum(coord[0] for coord in coordinates) / len(coordinates)
            avg_y = sum(coord[1] for coord in coordinates) / len(coordinates)

            if len(coordinates) >= 30:  # At least 30 frames for 1 second of movement at 30 FPS
                recent_positions = coordinates[-30:]  # Get the last 30 positions

                # Check for a valid bounce (descending followed by ascending)
                descending = False
                ascending = False
                bounce_detected = False

                for i in range(1, len(recent_positions)):
                    y1, y2 = recent_positions[i - 1][1], recent_positions[i][1]

                    if y1 > y2:  # Descending
                        descending = True
                    elif y1 < y2 and descending:  # Ascending after descending
                        ascending = True
                        if ascending and descending:
                            bounce_detected = True

                    if bounce_detected:
                        ball.bounces += 1
                        print(f"Bounce Detected for {ball.type.capitalize()}!")
                        bounce_detected = False  # Reset the bounce detection

            ball.prev_avg_y = avg_y

            cv2.putText(frame, f"Bounces {ball.type.capitalize()}: {ball.bounces}",
                        (20, 20 + balls.index(ball) * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.circle(frame, (int(avg_x), int(avg_y)), 5, (0, 255, 0), -1)

            ball.coordinates = []
            ball.timestamps = []


def main():
    model = initialize_yolo_model()
    cap = initialize_camera()
    balls = [Ball('basketball'), Ball('volleyball')]

    capture_interval = 1.0 / 30.0  # Capture a frame every 1/30th of a second
    start_time = time.time()

    while True:
        ret, frame = cap.read()

        detect_balls(frame, model, balls)

        elapsed_time = time.time() - start_time

        if elapsed_time >= 1.0:
            calculate_bounces(frame, balls)
            start_time = time.time()

        cv2.imshow('frame', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
