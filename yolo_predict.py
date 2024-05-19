from tabnanny import verbose
from ultralytics import YOLO
import cv2
import os
path = 'weights1/last.pt'
# image_path = 'sample_images/774555_0.jpg'
def yolo_prediction(image_path):

# Load a model
    model = YOLO(path)  # pretrained YOLOv8n model
    #img = cv2.imread(image_path)

    CONFTHRES = 0.5
    # Run batched inference on a list of images
    print("Image Path:", image_path)

    # Read the image from the specified path
    img = cv2.imread(image_path)

    # Check if img is None (failed to read image)
    if img is None:
        print("Error: Failed to read the image.")
        return None
    new_longest_side=1024
        # Get the dimensions of the image
    height, width = img.shape[:2]

    # Calculate the aspect ratio
    aspect_ratio = width / height

    # Determine the new dimensions based on the longest side
    if width > height:
        new_width = new_longest_side
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = new_longest_side
        new_width = int(new_height * aspect_ratio)

    # Resize the image
    resized_img = cv2.resize(img, (new_width, new_height))
    cv2.imwrite(image_path, resized_img)

    data_dict= {}
    results = model.predict(resized_img, conf=CONFTHRES, stream=False, device="cpu",verbose=False)#model([frame])
    #tempImage  = frame.copy()
    # Process results list

    for result in results:
        boxes = result.boxes
        confidences = boxes.conf

        for index, box in enumerate(boxes):
            if confidences[index].item() > CONFTHRES:
                class_name = results[0].names[int(box.cls)]
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates

                if class_name not in data_dict or confidences[index].item() > data_dict[class_name]["score"]:
                    data_dict[class_name] = {
                        "x": x1,
                        "y": y1,
                        "width": x2 - x1,
                        "height": y2 - y1,
                        "score": confidences[index].item()
                    }
                # tempImage = cv2.rectangle(tempImage,(int(x1), int(y1)),(int(x2), int(y2)), (0,0,255),thickness=2)
                # tempImage = cv2.putText(tempImage,class_name, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                """img = cv2.rectangle(resized_img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
                img = cv2.putText(img, f"{class_name}: {confidences[index].item():.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    output_image_path = os.path.join('outputs', 'output_image.jpg')
    cv2.imwrite(output_image_path, img)"""
    return data_dict
"""cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows() 
"""
