# Import necessary libraries
import cv2
from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os

# Declare model and padding variables
faceNet = cv2.dnn.readNet("utils/opencv_face_detector_uint8.pb", "utils/opencv_face_detector.pbtxt")
ageNet = cv2.dnn.readNet("utils/age_net.caffemodel", "utils/age_deploy.prototxt")
genderNet = cv2.dnn.readNet("utils/gender_net.caffemodel", "utils/gender_deploy.prototxt")
padding = 20

# Declare model mean values and lists
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes

def process_image(request):
    results = None
    uploaded_image = None
    noImage = False
    file_path = ""

    if request.method == 'POST':
        uploaded_file = request.FILES.get('uploaded_image', None)
        if uploaded_file:
            file_path = default_storage.save('image.jpg', ContentFile(uploaded_file.read()))
            file_path = os.path.join("media/", file_path)
            print(file_path, 'this is file path---------')

            frame = cv2.imread(file_path)
            resultImg, faceBoxes = highlightFace(faceNet, frame)

            if not faceBoxes:
                noImage = True

            results = []
            for faceBox in faceBoxes:
                face = frame[max(0, faceBox[1] - padding): min(faceBox[3] + padding, frame.shape[0] - 1),
                            max(0, faceBox[0] - padding): min(faceBox[2] + padding, frame.shape[1] - 1)]

                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                
                genderNet.setInput(blob)
                genderPreds = genderNet.forward()
                gender = genderList[genderPreds[0].argmax()]

                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                age = ageList[agePreds[0].argmax()][1:-1]

                results.append({'gender': gender, 'age': age})

    return render(request, 'upload_image.html', {'results': results, 'uploaded_image': file_path, 'noImage': noImage})
