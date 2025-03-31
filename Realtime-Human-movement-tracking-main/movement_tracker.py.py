import cv2
import mediapipe as mp
import numpy as np

class DetailedHumanTracker:
    def __init__(self):
        # Initialize MediaPipe solutions
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic

        # Track movement history for detailed analysis
        self.landmark_history = {
            'face': [],
            'left_hand': [],
            'right_hand': [],
            'pose': []
        }

    def get_landmark_coordinates(self, landmarks):
        """
        Extract detailed coordinates of specific landmarks
        """
        if not landmarks:
            return {}

        landmark_coords = {}
        
        # Face landmarks (total 468 landmarks)
        landmark_coords['face'] = {
            'nose_tip': (landmarks.landmark[1].x, landmarks.landmark[1].y),
            'left_eye': (landmarks.landmark[33].x, landmarks.landmark[33].y),
            'right_eye': (landmarks.landmark[263].x, landmarks.landmark[263].y),
            'left_ear': (landmarks.landmark[93].x, landmarks.landmark[93].y),
            'right_ear': (landmarks.landmark[323].x, landmarks.landmark[323].y),
            'mouth_left': (landmarks.landmark[78].x, landmarks.landmark[78].y),
            'mouth_right': (landmarks.landmark[308].x, landmarks.landmark[308].y)
        }
        
        return landmark_coords

    def track_movement(self):
        # Open webcam
        cap = cv2.VideoCapture(0)

        # MediaPipe Holistic model for comprehensive tracking
        with self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:
            
            while cap.isOpened():
                # Read frame from webcam
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                # Convert the BGR image to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                # Process the image and detect landmarks
                results = holistic.process(image)
                
                # Revert image color for display
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Comprehensive landmark drawing and tracking
                if results.face_landmarks:
                    # Draw face landmarks
                    self.mp_drawing.draw_landmarks(
                        image, 
                        results.face_landmarks, 
                        self.mp_holistic.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
                    )

                    # Extract and display face landmark coordinates
                    face_coords = self.get_landmark_coordinates(results.face_landmarks)
                    
                    # Display specific facial landmark coordinates
                    landmark_labels = [
                        f"Nose: ({face_coords['face']['nose_tip'][0]:.2f}, {face_coords['face']['nose_tip'][1]:.2f})",
                        f"Left Eye: ({face_coords['face']['left_eye'][0]:.2f}, {face_coords['face']['left_eye'][1]:.2f})",
                        f"Right Eye: ({face_coords['face']['right_eye'][0]:.2f}, {face_coords['face']['right_eye'][1]:.2f})"
                    ]
                    
                    # Display coordinates on screen
                    for i, label in enumerate(landmark_labels):
                        cv2.putText(
                            image, 
                            label, 
                            (10, 30 + i*30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, 
                            (0, 255, 0), 
                            2
                        )

                # Draw pose landmarks
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image, 
                        results.pose_landmarks, 
                        self.mp_holistic.POSE_CONNECTIONS
                    )

                # Draw hand landmarks
                if results.left_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image, 
                        results.left_hand_landmarks, 
                        self.mp_holistic.HAND_CONNECTIONS
                    )

                if results.right_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image, 
                        results.right_hand_landmarks, 
                        self.mp_holistic.HAND_CONNECTIONS
                    )

                # Display the image
                cv2.imshow('Detailed Human Tracking', image)

                # Break loop with 'q' key
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

def main():
    tracker = DetailedHumanTracker()
    print("Detailed Human Tracking Started!")
    print("Press 'q' to quit the application")
    tracker.track_movement()

if __name__ == "__main__":
    main()