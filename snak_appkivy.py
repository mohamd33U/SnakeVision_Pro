import cv2
from ultralytics import YOLO
import mysql.connector
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.screenmanager import ScreenManager, Screen


class ResultScreen(Screen):
    def __init__(self, predictions, species_info, annotated_image, **kwargs):
        super().__init__(**kwargs)
        self.predictions = predictions
        self.annotated_image = annotated_image

        layout = BoxLayout(orientation='vertical')

        # Label to display predictions
        self.prediction_label = Label(text="\n".join(predictions), size_hint_y=None, height=200)
        layout.add_widget(self.prediction_label)

        # Label to display species info
        self.species_info_label = Label(text="\n".join(species_info), size_hint_y=None, height=200)
        layout.add_widget(self.species_info_label)

        # Display the annotated image
        self.result_image = Image()
        self.update_result_image(self.annotated_image)
        layout.add_widget(self.result_image)

        # Back button to return to the camera screen
        back_button = Button(text='Back to Camera', size_hint_y=None, height=50)
        back_button.bind(on_press=self.back_to_camera)
        layout.add_widget(back_button)

        self.add_widget(layout)

    def update_result_image(self, image):
        buf = cv2.flip(image, 0)
        buf = cv2.cvtColor(buf, cv2.COLOR_BGR2RGBA)
        texture = Texture.create(size=(buf.shape[1], buf.shape[0]), colorfmt='rgba')
        texture.blit_buffer(buf.tobytes(), colorfmt='rgba', bufferfmt='ubyte')
        self.result_image.texture = texture

    def back_to_camera(self, instance):
        self.manager.current = 'camera'  # Switch back to the camera screen


class CameraScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical')

        # Image widget to display the camera feed
        self.image_widget = Image()
        self.layout.add_widget(self.image_widget)

        # Capture button to take a picture
        capture_btn = Button(text='Capture Image', size_hint_y=None, height=50)
        capture_btn.bind(on_press=self.capture_image)
        self.layout.add_widget(capture_btn)

        self.add_widget(self.layout)

        # Initialize camera
        self.capture = cv2.VideoCapture(0)
        self.model = YOLO('models/best_snake.pt')  # Load the YOLO model once
        Clock.schedule_interval(self.update_frame, 1.0 / 30.0)  # 30 FPS

    def update_frame(self, dt):
        ret, frame = self.capture.read()
        if ret:
            # Convert frame to texture for Kivy
            buf = cv2.flip(frame, 0)
            buf = cv2.cvtColor(buf, cv2.COLOR_BGR2RGBA)
            texture = Texture.create(size=(buf.shape[1], buf.shape[0]), colorfmt='rgba')
            texture.blit_buffer(buf.tobytes(), colorfmt='rgba', bufferfmt='ubyte')
            self.image_widget.texture = texture

    def capture_image(self, instance):
        ret, frame = self.capture.read()
        if ret:
            self.process_image(frame)  # Process the captured frame

    def process_image(self, image):
        # Predict using YOLO
        results = self.model.predict(image, save=False, conf=0.25)
        result = results[0]

        predictions = []
        class_names = []  # To store class names for database updates
        species_info = []

        for r in result.boxes:
            class_id = int(r.cls)
            confidence = r.conf.item()
            class_name = self.model.names[class_id] if self.model.names else f"Class {class_id}"

            # Format the class name for database querying
            formatted_species_name = "".join(word.capitalize() for word in class_name.split("_"))
            class_names.append(formatted_species_name)  # Save formatted name for later use

            predictions.append(f"Detected: {class_name}, Confidence: {confidence:.2f}")

        # Visualize results
        annotated_image = result.plot()

        # Database connection
        config = {
            'user': 'root',
            'password': 'CD445325',
            'host': 'localhost',
            'database': 'snake_database',
        }

        try:
            connection = mysql.connector.connect(**config)
            cursor = connection.cursor(dictionary=True)

            for species_name in class_names:
                query = """
                    SELECT Id, species_name, venomous_status, first_aid, countries_found
                    FROM snakes
                    WHERE species_name = %s
                """
                cursor.execute(query, (species_name,))
                row = cursor.fetchone()
                if row:
                    species_info.append(
                        f"Species: {row['species_name']}, "
                        f"Venomous: {row['venomous_status']}, "
                        f"First Aid: {row['first_aid']}, "
                        f"Countries Found: {row['countries_found']}"
                    )
                else:
                    species_info.append(f"No data found for species: {species_name}")

            result_screen = ResultScreen(predictions, species_info, annotated_image, name='result')
            self.manager.add_widget(result_screen)
            self.manager.current = 'result'  # Switch to the result screen

        except mysql.connector.Error as err:
            print(f"Database Error: {err}")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            if 'connection' in locals() and connection.is_connected():
                cursor.close()
                connection.close()
                print("Connection closed")

    def on_stop(self):
        if self.capture.isOpened():
            self.capture.release()  # Release the camera on exit


class SnakeApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(CameraScreen(name='camera'))  # Add camera screen
        return sm


if __name__ == '__main__':
    SnakeApp().run()
