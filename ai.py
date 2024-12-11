from PIL import Image as PilImage  
import cv2
import numpy as np
import gradio as gr


net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")  
layer_names = net.getLayerNames()
output_layers = net.getUnconnectedOutLayersNames()


classes = []
with open("coco.names", "r") as f:  
    classes = [line.strip() for line in f.readlines()]

def detect_objects(input_image):
   
    img = np.array(input_image) 
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

   
    class_ids = []
    confidences = []
    boxes = []
    object_counts = {}  

   
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  
        
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)


                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)    

    result_image = img.copy()
    final_object_counts = {}  
    for i in range(len(boxes)):  
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)  # BGR format
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(result_image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            

            if label in final_object_counts:
                final_object_counts[label] += 1
            else:
                final_object_counts[label] = 1

    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

    object_count_summary = "\n".join([f"{key}: {value}" for key, value in final_object_counts.items()])

    return result_image, object_count_summary


iface = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(type="numpy"), gr.Textbox()],
    live=True
)

iface.launch()
