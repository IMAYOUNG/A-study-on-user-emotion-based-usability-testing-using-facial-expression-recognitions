import cv2
from deepface import DeepFace
import time
import csv

def detect_emotion(video_obj, output_csv):
    # Load face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Start capturing video
    cap = cv2.VideoCapture(0)

    happy_cnt = 0
    sad_cnt = 0
    neutral_cnt = 0
    disgust_cnt = 0
    angry_cnt = 0
    fear_cnt = 0
    surprise_cnt = 0

    # Start time
    start_time_total = time.time()
    start_time = time.strftime('%H:%M:%S')

    # Open CSV file for writing
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time', 'Predicted Emotion'])

        while video_obj.video_active:
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Convert frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Convert grayscale frame to RGB format
            rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100))

            for (x, y, w, h) in faces:
                # Extract the face ROI (Region of Interest)
                face_roi = rgb_frame[y:y + h, x:x + w]

                # Perform emotion analysis on the face ROI
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

                 # End time for emotion analysis
                end_time_analysis = time.time()

                # Get the dominant emotion and its probability
                emotion = result[0]['dominant_emotion']
                emotion_prob = result[0]['emotion'][emotion]

                # Check if the emotion probability exceeds the threshold
                if emotion_prob >= 0.9:
                    if emotion == 'happy':
                        happy_cnt += 1
                    elif emotion == 'sad':
                        sad_cnt += 1
                    elif emotion == 'angry':
                        angry_cnt += 1
                    elif emotion == 'fear':
                        fear_cnt += 1
                    elif emotion == 'disgust':
                        disgust_cnt += 1
                    elif emotion == 'surprise':
                        surprise_cnt += 1
                    elif emotion == 'neutral':
                        neutral_cnt += 1
                else:
                    emotion = 'neutral'  # If below threshold, classify as neutral

                # Write data to CSV
                writer.writerow([time.strftime('%H:%M:%S'), emotion])

                # Draw rectangle around face and label with predicted emotion
                if emotion in ['sad', 'angry', 'fear', 'disgust']:
                    rectangle_color = (0, 0, 255)  # Red for negative emotions
                    text_color = (0, 0, 255)  # Red for negative emotions
                elif emotion in ['neutral', 'surprise']:
                    rectangle_color = (255, 255, 255)
                    text_color = (255, 255, 255)
                elif emotion in ['happy']:
                    rectangle_color = (0, 255, 0)  # Green for positive emotions
                    text_color = (0, 255, 0)  # Green for positive emotions

                # Draw rectangle around face and label with predicted emotion
                cv2.rectangle(frame, (x, y), (x + w, y + h), rectangle_color, 2)
                cv2.putText(frame, f"{emotion} ", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

            # Display the resulting frame
            cv2.imshow('Real-time Emotion Detection', frame)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # End time
        end_time_total = time.time()
        end_time = time.strftime('%H:%M:%S')

        # Total execution time
        total_execution_time = end_time_total - start_time_total

        # Write additional information to CSV
        writer.writerow([''])
        writer.writerow([''])
        writer.writerow(['Start Time', 'End Time', 'Total Execution Time'])
        writer.writerow([start_time, end_time, total_execution_time])
        
        writer.writerow(['Happy cnt', 'Sad cnt', 'Neutral cnt', 'Angry cnt', 'Fear cnt', 'Disgust cnt', 'Surprise cnt'])
        writer.writerow([happy_cnt, sad_cnt, neutral_cnt, angry_cnt, fear_cnt, disgust_cnt, surprise_cnt])
        
    # Release the capture and close all windows
    cap.release()
    cv2.destroyAllWindows()
    
    # Print counts of each emotion and execution time
    print("Happy count:", happy_cnt, "// Sad count:", sad_cnt, "// Neutral count:", neutral_cnt)
    print("Angry count:", angry_cnt, "// Fear count:", fear_cnt, "// Disgust count:", disgust_cnt, "// Surprise count:", surprise_cnt )
    print("** Start time:", start_time, "// End time:", end_time)
    print("** Total execution time:", total_execution_time, "seconds")

# import cv2
# from deepface import DeepFace
# import time
# import csv

# def detect_emotion(video_obj, output_csv):
#     # Load face cascade classifier
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     # Start capturing video
#     cap = cv2.VideoCapture(0)

#     happy_cnt = 0
#     sad_cnt = 0
#     neutral_cnt = 0
#     disgust_cnt = 0
#     angry_cnt = 0
#     fear_cnt = 0
#     surprise_cnt = 0

#     # Start time
#     start_time_total = time.time()
#     start_time = time.strftime('%H:%M:%S')

#     # Open CSV file for writing
#     with open(output_csv, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['Time', 'Predicted Emotion'])

#         while video_obj.video_active:
#             # Capture frame-by-frame
#             ret, frame = cap.read()

#             # Convert frame to grayscale
#             gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#             # Convert grayscale frame to RGB format
#             rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

#             # Detect faces in the frame
#             faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.05, minNeighbors=10, minSize=(80, 80))

#             for (x, y, w, h) in faces:
#                 # Extract the face ROI (Region of Interest)
#                 face_roi = rgb_frame[y:y + h, x:x + w]

#                 # Start time for emotion analysis
#                 start_time_analysis = time.time()

#                 # Perform emotion analysis on the face ROI
#                 result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

#                 # End time for emotion analysis
#                 end_time_analysis = time.time()

#                 # Determine the dominant emotion
#                 emotion = result[0]['dominant_emotion']
                
#                 if emotion in ['happy']:
#                     happy_cnt += 1
#                 elif emotion in ['sad']:
#                     sad_cnt += 1
#                 elif emotion in ['angry']:
#                     angry_cnt += 1
#                 elif emotion in ['fear']:
#                     fear_cnt += 1
#                 elif emotion in ['disgust']:
#                     disgust_cnt += 1
#                 elif emotion in ['surprise']:
#                     surprise_cnt += 1
#                 elif emotion in ['neutral']:
#                     neutral_cnt += 1

#                 # Write data to CSV
#                 writer.writerow([time.strftime('%H:%M:%S'), emotion]) 
#                                 #  execution_time_analysis])

#                 # Draw rectangle around face and label with predicted emotion
#                 if emotion in ['sad', 'angry', 'fear', 'disgust']:
#                     rectangle_color = (0, 0, 255)  # Red for negative emotions
#                     text_color = (0, 0, 255)  # Red for negative emotions
#                 elif emotion in ['neutral', 'surprise']:
#                     rectangle_color = (255, 255, 255)  
#                     text_color = (255, 255, 255)  
#                 elif emotion in ['happy']:
#                     rectangle_color = (0, 255, 0)  # Green for positive emotions
#                     text_color = (0, 255, 0)  # Green for positive emotions

#                 # Draw rectangle around face and label with predicted emotion
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), rectangle_color, 2)
#                 cv2.putText(frame, f"{emotion} ", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
#                                                 # ({execution_time_analysis:.2f}s)"
#                 # Print the predicted emotion with execution time for each face
#                 print(f"{time.strftime('%H:%M:%S')} Predicted Emotion: {emotion} ")
#                 # (Execution Time: {execution_time_analysis:.2f} seconds)")

#             # Display the resulting frame
#             cv2.imshow('Real-time Emotion Detection', frame)

#             # Press 'q' to exit
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         # End time
#         end_time_total = time.time()
#         end_time = time.strftime('%H:%M:%S')

#         # Total execution time
#         total_execution_time = end_time_total - start_time_total

#         writer.writerow([''])
#         writer.writerow([''])
#         writer.writerow(['Start Time', 'End Time', 'Total Execution Time'])
#         writer.writerow([start_time, end_time, total_execution_time])
        
#         writer.writerow(['Happy cnt', 'Sad cnt', 'Neutral cnt', 'Angry cnt', 'Fear cnt', 'Disgust cnt', 'Surprise cnt'])
#         writer.writerow([happy_cnt, sad_cnt, neutral_cnt, angry_cnt, fear_cnt, disgust_cnt, surprise_cnt])
        
#     # Release the capture and close all windows
#     cap.release()
#     cv2.destroyAllWindows()
    
#     print("Happy count:", happy_cnt, "// Sad count:", sad_cnt, "// Neutral count:", neutral_cnt)
#     print("Angry count:", angry_cnt, "// Fear count:", fear_cnt, "// Disgust count:", disgust_cnt, "// Surprise count:", surprise_cnt )
    
#     # Print start time, end time, and total execution time
#     print("** Start time:", start_time, "// End time:", end_time)
#     print("** Total execution time:", total_execution_time, "seconds")