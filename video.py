import cv2
import numpy as np

net = cv2.dnn.readNet('yolov3.weights' , 'yolov3.cfg')

classes = []


with open('coco.names' , 'r') as f:
    classes = f.read().splitlines()
    

file = input("Enter the filepath : ")
vid = cv2.VideoCapture(file)


codec = cv2.VideoWriter_fourcc(*'XVID')
vid_fps =  int(vid.get(cv2.CAP_PROP_FPS))
vid_width , vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)) , int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
out= cv2.VideoWriter('output.avi' , codec , vid_fps , (vid_width,vid_height))


while True:
    _ , image = vid.read()

    height , width , _ = image.shape

    # converting the image to specfic form by that we can pass this as input to the model
    blob = cv2.dnn.blobFromImage(image , 1/255 , (416,416) ,(0,0,0) , swapRB = True , crop = False)

    net.setInput(blob)
    output_layer_names = net.getUnconnectedOutLayersNames()
    layeroutputs = net.forward(output_layer_names)

    #visualize
    #Extract the bounding boxes and confidences and predicted classes
    boxes=[]
    confidences=[]
    class_id=[]

    for output in layeroutputs:
        for detection in output:
            score = detection[5:]
            ids = np.argmax(score)
            confidence = score[ids]
            if confidence > 0.5: 
                center_x = int(detection[0]*width)# to denoramalize multiplying with original h and w
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_id.append(ids)

    # To avoid many boxes we are picking up the boxes with high confidence
    indexes = cv2.dnn.NMSBoxes(boxes,confidences ,0.5,0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0,255,size=(len(boxes), 3))


    if len(indexes)>0:
        for i in indexes.flatten():
            x,y,w,h = boxes[i]
            label = str(classes[class_id[i]])
            confi = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(image , (x,y) , (x+w,y+h) , color ,2)
            cv2.putText(image , label +" "+confi , (x,y+20) ,font ,2 ,(255,255,255),2)

    cv2.imshow("Frame" , image)
    out.write(image)
    key = cv2.waitKey(1)
    if key == 27:
        break
vid.release()
out.release()
cv2.destroyAllWindows()
