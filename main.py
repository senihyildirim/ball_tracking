import cv2
import time
from ultralytics import YOLO


# Define a class to represent a ball
class Ball:
    def __init__(self, ball_type, display_name):
        self.type = ball_type  # Type of the ball (e.g., "tennis", "soccer")
        self.display_name = display_name  # A unique name for the ball
        self.prev_avg_y = None  # Previous average Y-coordinate of the ball
        self.bouncing = False  # Flag indicating whether the ball is bouncing
        self.bounces = 0  # Number of bounces detected for the ball
        self.coordinates = []  # List to store ball coordinates
        self.timestamps = []  # List to store timestamps


# Initialize the YOLO object detection model
def initialize_yolo_model():
    return YOLO('best.pt')


# Initialize the camera capture object
def initialize_camera():
    # cap = cv2.VideoCapture("multiple_balls.mp4") #for multiple balls
    cap = cv2.VideoCapture("basketball.mp4")  # for single ball
    cap.set(3, 800)  # Set video width
    cap.set(4, 400)  # Set video height
    return cap


# Detect balls in a frame and track them
def detect_balls(frame, model, balls):
    results = model.track(frame, persist=True)

    for result in results:
        ball_type = None
        for name in result.names.values():
            if name not in [ball.type for ball in balls]:
                ball_type = name
                break

        if ball_type is not None:
            # Create a new Ball object when a new ball is detected
            new_ball = Ball(ball_type, f'ball_{len(balls) + 1}')
            balls.append(new_ball)

        for ball in balls:
            if ball.type in result.names.values():
                ball_indices = [i for i, name in result.names.items() if name == ball.type]

                for idx in ball_indices:
                    if len(result.boxes) > idx:
                        bbox = result.boxes[idx]
                        x1, y1, x2, y2 = bbox.xyxy[0].tolist()

                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)

                        # Update ball coordinates and timestamps
                        ball.coordinates.append((center_x, center_y))
                        ball.timestamps.append(time.time())

                        # Draw a circle and label for the ball on the frame
                        cv2.circle(frame, (center_x, center_y), 20, (0, 0, 255), -1)
                        cv2.putText(frame, f'{ball.display_name}',
                                    (center_x - 15, center_y + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


# Calculate bounces for each ball
def calculate_bounces(frame, balls):
    for ball in balls:
        coordinates = ball.coordinates
        timestamps = ball.timestamps
        if len(coordinates) >= 3:
            y1, y2, y3 = coordinates[-3:]

            descending = y1[1] > y2[1] and y2[1] > y3[1]
            ascending = y1[1] < y2[1] and y2[1] < y3[1]

            if descending and ball.bouncing == False:
                ball.bouncing = True
                ball.bounces += 1
            elif ascending and ball.bouncing == True:
                ball.bouncing = False

            # Display the number of bounces for the ball on the frame
            cv2.putText(frame, f"Bounces {ball.display_name}: {ball.bounces}",
                        (20, 20 + balls.index(ball) * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


# Main function
def main():
    model = initialize_yolo_model()  # Initialize YOLO model
    cap = initialize_camera()  # Initialize camera capture
    balls = []  # List to store detected balls

    start_time = time.time()

    while True:
        ret, frame = cap.read()  # Read a frame from the camera


        detect_balls(frame, model, balls)  # Detect and track balls in the frame

        elapsed_time = time.time() - start_time

        if elapsed_time >= 1.0:
            calculate_bounces(frame, balls)  # Calculate bounces
            start_time = time.time()

        cv2.imshow('frame', frame)  # Display the frame

        if cv2.waitKey(25) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()  # Release the camera
    cv2.destroyAllWindows()  # Close the OpenCV window


if __name__ == "__main__":
    main()
