import cv2
from ultralytics import YOLO
import mysql.connector

# Load a pre-trained YOLOv8 model
model = YOLO('models/best_snake.pt')  # Ensure the correct model path

# Load an image
image_path = 'AcanthophisAntarcticusNonVenomous/3.jpeg'
image = cv2.imread(image_path)

# Ensure the image is loaded correctly
if image is None:
    print(f"Failed to load image at {image_path}")
else:
    # Predict
    results = model.predict(image, save=False, conf=0.25)  # Adjust confidence threshold if needed

    # Get the first result (if multiple images, take the first one)
    result = results[0]

    # Print class names and confidence scores
    for r in result.boxes:  # Loop over each detected box
        class_id = int(r.cls)  # Class ID (integer)
        confidence = r.conf.item()  # Convert confidence from tensor to float

        # You can access class names from model.names, if available
        class_name = model.names[class_id] if model.names else f"Class {class_id}"

        print(f"Detected: {class_name}, Confidence: {confidence:.2f}")

    # Visualize the results
    annotated_image = result.plot()  # Plot the boxes and labels on the image

    # Replace with your database connection details
    config = {
        'user': 'your',
        'password': 'your',
        'host': 'your',
        'database': 'your',  # Ensure this database exists
    }

    try:
        connection = mysql.connector.connect(**config)

        if connection.is_connected():
            print("Successfully connected to the database")

            cursor = connection.cursor(dictionary=True)  # Use dictionary for named columns
            query = """
                SELECT Id, species_name, venomous_status, first_aid, countries_found
                FROM snakes  # Update this to the actual table name
                WHERE species_name = %s
            """

            # Transform class_name into desired format
            spname = "".join(word.capitalize() for word in class_name.split("_"))
            desired_species_name = spname  # Set the desired species name

            # Execute the query
            cursor.execute(query, (desired_species_name,))

            # Fetch the result
            row = cursor.fetchone()
            if row:
                print("Retrieved row:", row)
            else:
                print("No row found with the specified species_name.")

    except mysql.connector.Error as err:
        print(f"Error: {err}")

    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()
            print("Connection closed")

    # Display the image with predictions
    cv2.imshow('YOLOv8 Predictions', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
