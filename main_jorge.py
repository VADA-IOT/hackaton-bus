# -*- coding: utf-8 -*-

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import imutils
import pytesseract
from PIL import Image
import time
from retina.utils import visualize_boxes

INPUT_IMAGE = False

MODEL_PATH = 'snapshots/resnet50_full.h5'

# 'path to input image/video'
#IMAGE='./4.jpg'
IMAGE='./real_photos/frame0.jpg'
VIDEO = './real_videos/seoul1.mp4'

# 'path to yolo config file' 
# download https://github.com/arunponnusamy/object-detection-opencv/blob/master/yolov3.cfg
CONFIG='./yolov3.cfg'

# 'path to text file containing class names'
# download https://github.com/arunponnusamy/object-detection-opencv/blob/master/yolov3.txt
CLASSES='./yolov3.txt'

# 'path to yolo pre-trained weights' 
# wget https://pjreddie.com/media/files/yolov3.weights
WEIGHTS='./yolov3.weights'



import os  
print(os.path.exists(CLASSES))
print(os.path.exists(CONFIG))
print(os.path.exists(WEIGHTS))
print(os.path.exists(IMAGE))
print(os.path.exists(VIDEO))

# read class names from text file
classes = None
with open(CLASSES, 'r') as f:
     classes = [line.strip() for line in f.readlines()]
        
scale = 0.00392
conf_threshold = 0.5
nms_threshold = 0.4

# generate different colors for different classes 
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# function to get the output layer names 
# in the architecture

def load_inference_model(model_path=os.path.join('snapshots', 'resnet.h5')):
    model = models.load_model(model_path, backbone_name='resnet50')
    model = models.convert_model(model)
    model.summary()
    return model

def post_process(boxes, original_img, preprocessed_img):
    # post-processing
    h, w, _ = preprocessed_img.shape
    h2, w2, _ = original_img.shape
    boxes[:, :, 0] = boxes[:, :, 0] / w * w2
    boxes[:, :, 2] = boxes[:, :, 2] / w * w2
    boxes[:, :, 1] = boxes[:, :, 1] / h * h2
    boxes[:, :, 3] = boxes[:, :, 3] / h * h2
    return boxes

def get_output_layers(net): 
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def brigthness_level(img):
    return img.mean(axis=0).mean()
    
# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def processImage(image,index):

    Width = image.shape[1]
    Height = image.shape[0]
    print(Width);
    print(Height);
    
    #cv2.imshow('original frame',image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    # read pre-trained model and config file
    net = cv2.dnn.readNet(WEIGHTS, CONFIG)

    # create input blob 
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
    #blob = cv2.dnn.blobFromImage(image, scale, (320,320), (0,0,0), True, crop=False)
    # set input blob for the network
    net.setInput(blob)

    # run inference through the network
    # and gather predictions from output layers
    outs = net.forward(get_output_layers(net))

    # initialization
    class_ids = []
    confidences = []
    boxes_b = []
    # for each detetion from each output layer 
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.70:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes_b.append([x, y, w, h])
                
                # # do OCR in half upper part of the boxes
                # print("la parte donde esta mirando el testo de las imagenes");
                # cropped_img = image[y:y + h, x:x + w]  #--- Notice this part where you have to add the stride as well ---
                # cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY) #convert to grey scale
                # cropped_img= cv2.bilateralFilter(cropped_img, 11, 17, 17) #Blur to reduce noise
                # cv2.imshow('img',cropped_img)
                # custom_config = r'--psm 11'
                # text = pytesseract.image_to_string(cropped_img,lang="digits_added", config= custom_config)
                # print("Cropped:",text)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
            
    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes_b, confidences, conf_threshold, nms_threshold)

    # go through the detections remaining
    # after nms and draw bounding box
    for i in indices:
        i = i[0]
        box = boxes_b[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        #print("boxes to be drawn");
        draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        # cv2.imshow("object_location_boxes", image)
        # # #wait until any key is pressed
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        
        if (class_ids[i] == 5):
            print('bus located');
            # do OCR in half upper part of the boxes
            print("la parte donde esta mirando el testo de las imagenes");
            cropped_img = image[int(y):int(y) + int(h/3), int(x) + int(1*w/10):int(x) + int(7*w/10)]  #--- Notice this part where you have to add the stride as well ---
            #cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY) #convert to grey scale
            #cropped_img= cv2.bilateralFilter(cropped_img, 11, 17, 17) #Blur to reduce noise
            cv2.imshow('img',cropped_img)

            # copy to draw on
            draw = cropped_img.copy()
            draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    
            # # preprocess image for network
            cropped_img = preprocess_image(cropped_img)
            cropped_img, _ = resize_image(cropped_img, 416, 448)
            # #cropped_img, _ = resize_image(cropped_img, 410, 448)
            plt.imshow(cropped_img);
            #plt.imshow(_);
    
            # process image
            start = time.time()
            boxes_d, scores, labels = model.predict_on_batch(np.expand_dims(cropped_img, axis=0))
            print(labels);
            print("processing time: ", time.time() - start)
    
            boxes_d = post_process(boxes_d, draw, cropped_img)
            labels = labels[0]
            scores = scores[0]
            boxes_d = boxes_d[0]
            print(labels.shape)
            print(boxes_d.shape)
            print(scores[0])
                        
    
     
            visualize_boxes(draw, boxes_d, labels, scores, class_labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    
            # 5. plot    
            plt.imshow(draw)
            plt.show()
           
            
            digits_array = np.concatenate((boxes_d, scores[:,np.newaxis], labels[:,np.newaxis]), axis=1)
            digits_array = digits_array[digits_array[:,4] > 0.35]
            sortedDigits_score = digits_array[digits_array[:,0].argsort()] 
            #sortedDigits_left_right = digits_array[digits_array[:,0].argsort()]
            print(sortedDigits_score)
            ##### UNcomment when I solve the errors when it does not get three numbers and also when you solve the algorithm for knowing how many numbers there are.
            # print(sortedDigits_score.shape)
            # print("the number is")
            # print(str(int(sortedDigits_score[0][5])) + str(int(sortedDigits_score[1][5])) + str(int(sortedDigits_score[2][5])))
            
            
            # #grayscaled = cv2.cvtColor(cropped_img,cv2.COLOR_BGR2GRAY)
            # #brigthness = brigthness_level(grayscaled);
            # #print(brigthness);
            # #retval, th = cv2.threshold(grayscaled, brigthness*1.2, 255, cv2.THRESH_BINARY_INV)
            # #th = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
            # #retval2,th = cv2.threshold(grayscaled,254,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            # #cv2.imshow('Adaptive threshold',th)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # print("to print image to data");
            # print(pytesseract.image_to_data(Image.fromarray(cropped_img), config = '--psm 11',output_type='data.frame'))
            # custom_config = r'--psm 11'
            # text = pytesseract.image_to_string(cropped_img,lang="digits_added", config= custom_config)
            # print("Cropped:",text)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    # # display output image    
    # out_image_name = "object detection"+str(index)
    #cv2.imshow("object_location_boxes", image)
    # #wait until any key is pressed
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    # # save output image to disk
    # print("image to be saved");
    # cv2.imwrite("./out/"+out_image_name+".jpg", image) 
    
    



#open the image

model = load_inference_model(MODEL_PATH)

if (INPUT_IMAGE == True):
    frame = cv2.imread(IMAGE)
    print(type(frame))
    framecp = frame.copy();
    framecp = cv2.resize(framecp,None, fx=0.6, fy=0.6) 
    assert not isinstance(framecp,type(None)), 'image not found'
    index=0;
    processImage(framecp, index);

else:
    # Opens the Video file
    cap = cv2.VideoCapture(VIDEO)
    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            print('no frame')
            break
        print(i)
        framecp = frame.copy()
        print(type(frame))
        framecp = frame.copy();
        #framecp = cv2.resize(framecp,None, fx=0.6, fy=0.6) 
        framecp = cv2.resize(framecp,None, fx=1, fy=1)
        # Width = framecp.shape[1]
        # Height = framecp.shape[0]
        # center = (Width / 2, Height / 2)
        # angle270 = 270
        ####AMAZING!! THIS COMMENTED LINE THAT IS ONLY USED IF I ROTATE AND THAT SUPPOSELY SHOULD NOT AFFECT ANYTHING DESTROYS MAKES THE OBJECT LOCATION ALGORITHM CRAZY
        ########## ONLY IF ONLY I NAME IT "SCALE"....
        #scale_2=1.0
        ###################################################################################################################################################################
        #cv2.imwrite('frame'+str(i)+'.jpg',frame)
        #M = cv2.getRotationMatrix2D(center, angle270, scale_2)
        #image = cv2.warpAffine(framecp, M, (Height, Width)) 
        #framecp = np.rot90(framecp, 3)
        processImage(framecp, i)
        i+=1

    cap.release()



cv2.destroyAllWindows()


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#img = mpimg.imread('out/object detection1.jpg')
#plt.imshow(img)




