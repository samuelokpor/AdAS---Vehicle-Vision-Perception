# import torchvision
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from PIL import Image
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import torch


# COCO_INSTANCE_CATEGORY_NAMES = [
#     '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#     'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
#     'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#     'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
#     'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#     'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
#     'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
#     'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#     'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
#     'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
#     'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
#     'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
# ]

# # Load a model pre-trained on COCO
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# # For inference, switch to eval mode
# model.eval()

# # Load an image
# image = Image.open('data/test3.png').convert("RGB")

# # Transform the image
# transform = torchvision.transforms.Compose([
#     torchvision.transforms.ToTensor() 
# ])
# image_tensor = transform(image)

# # Add a batch dimension
# image_tensor = image_tensor.unsqueeze(0)

# # Perform the detection
# with torch.no_grad():
#     prediction = model(image_tensor)

# # The output is a list of dict, where each dict contains the predicted
# # classes, boxes and scores for each image in the batch.
# # We only have one image in the batch so we take the first item in the list
# prediction = prediction[0]

# # Create a figure and a set of subplots
# fig, ax = plt.subplots(1)

# # Display the image
# ax.imshow(image)

# # Print the prediction
# for i in range(len(prediction['boxes'])):
#     x1, y1, x2, y2 = map(int, prediction['boxes'][i])
#     label = int(prediction['labels'][i])
#     score = float(prediction['scores'][i])

#     # Only consider detections with a confidence score of 0.9 or higher
#     if score >= 0.9:
#         label_name = COCO_INSTANCE_CATEGORY_NAMES[label]
#         print(f"Detected a {label_name} with confidence {score} at {x1}, {y1}, {x2}, {y2}")

#         # Create a Rectangle patch
#         rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='g', facecolor='none')

#         # Add the patch to the Axes
#         ax.add_patch(rect)

#         plt.text(x1, y1, f"{label_name} ", color="red")

# # Show the figure with the bounding boxes
# plt.show()


import cv2
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.patches as patches
import numpy as np
import torch

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# Load a model pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

# Open the video file
cap = cv2.VideoCapture('data/drive_test.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        # Convert the frame to PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Transform the image
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor() 
        ])
        image_tensor = transform(image)

        # Add a batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device)

        # Perform the detection
        with torch.no_grad():
            prediction = model(image_tensor)

        # The output is a list of dict, where each dict contains the predicted
        # classes, boxes and scores for each image in the batch.
        # We only have one image in the batch so we take the first item in the list
        prediction = prediction[0]

        # Draw bounding boxes and labels on the frame
        for i in range(len(prediction['boxes'])):
            x1, y1, x2, y2 = map(int, prediction['boxes'][i])
            label = int(prediction['labels'][i])
            score = float(prediction['scores'][i])

            # Only consider detections with a confidence score of 0.9 or higher
            if score >= 0.9:
                label_name = COCO_INSTANCE_CATEGORY_NAMES[label]

                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw the label
                cv2.putText(frame, f"{label_name}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        # Display the frame with bounding boxes
        cv2.imshow('frame', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

# Release everything when done
cap.release()
cv2.destroyAllWindows()






