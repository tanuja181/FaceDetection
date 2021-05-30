from django.apps import AppConfig


class DetectionConfig(AppConfig):
    name = 'detection'


# import cv2
# import math
# import argparse
#
#
# def highlightFace(net, frame, conf_threshold=0.7):
#     frameOpencvDnn = frame.copy()
#     frameHeight = frameOpencvDnn.shape[0]
#     frameWidth = frameOpencvDnn.shape[1]
#     blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
#
#     net.setInput(blob)
#     detections = net.forward()
#     faceBoxes = []
#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > conf_threshold:
#             x1 = int(detections[0, 0, i, 3] * frameWidth)
#             y1 = int(detections[0, 0, i, 4] * frameHeight)
#             x2 = int(detections[0, 0, i, 5] * frameWidth)
#             y2 = int(detections[0, 0, i, 6] * frameHeight)
#             faceBoxes.append([x1, y1, x2, y2])
#             cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
#     return frameOpencvDnn, faceBoxes
#
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--image')
#
# args = parser.parse_args()
#
# faceProto = "opencv_face_detector.pbtxt"
# faceModel = "opencv_face_detector_uint8.pb"
# ageProto = "age_deploy.prototxt"
# ageModel = "age_net.caffemodel"
# genderProto = "gender_deploy.prototxt"
# genderModel = "gender_net.caffemodel"
#
# MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# '''ageList=['(1)','(2)','(3)','(4)','(5)','(6)','(7)','(8)','(9)','(10)',
#          '(11)','(12)','(13)','(14)','(15)','(16)','(17)','(18)','(19)','(20)',
#          '(21)','(22)','(23)','(24)','(25)','(26)','(27)','(28)','(29)','(30)',
#          '(31)','(32)','(33)','(34)','(35)','(36)','(37)','(38)','(39)','(40)',
#          '(41)','(42)','(43)','(44)','(45)','(46)','(47)','(48)','(49)','(50)',
#          '(51)','(52)','(53)','(54)','(55)','(56)','(57)','(58)','(59)','(60)',
#          '(61)','(62)','(63)','(64)','(65)','(66)','(67)','(68)','(69)','(70)',
#          '(71)','(72)','(73)','(74)','(75)','(76)','(77)','(78)','(79)','(80)',
#          '(81)','(82)','(83)','(84)','(85)','(86)','(87)','(88)','(89)','(90)',
#          '(91)','(92)','(93)','(94)','(95)','(96)','(97)','(98)','(99)','(100)']
# '''
# genderList = ['Male', 'Female']
#
# faceNet = cv2.dnn.readNet(faceModel, faceProto)
# ageNet = cv2.dnn.readNet(ageModel, ageProto)
# genderNet = cv2.dnn.readNet(genderModel, genderProto)
#
# video = cv2.VideoCapture(args.image if args.image else 0)
# padding = 20
# while cv2.waitKey(1) < 0:
#     hasFrame, frame = video.read()
#     if not hasFrame:
#         cv2.waitKey()
#         break
#
#     resultImg, faceBoxes = highlightFace(faceNet, frame)
#     if not faceBoxes:
#         print("No face detected")
#
#         # predict ages and genders of the detected faces
#         results = model.predict(faces)
#         predicted_genders = results[0]
#         ages = np.arange(0, 101).reshape(101, 1)
#         predicted_ages = results[1].dot(ages).flatten()
#
#         # draw results
#         for i, d in enumerate(detected):
#             label = "{}, {}".format(int(predicted_ages[i]),
#                                     "M" if predicted_genders[i][0] < 0.5 else "F")
#             draw_label(img, (d.left(), d.top()), label)
#
#     for faceBox in faceBoxes:
#         face = frame[max(0, faceBox[1] - padding):
#                      min(faceBox[3] + padding, frame.shape[0] - 1), max(0, faceBox[0] - padding)
#                                                                     :min(faceBox[2] + padding, frame.shape[1] - 1)]
#         blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
#         genderNet.setInput(blob)
#         genderPreds = genderNet.forward()
#         gender = genderList[genderPreds[0].argmax()]
#         print(f'Gender: {gender}')
#
#         ageNet.setInput(blob)
#         agePreds = ageNet.forward()
#         age = ageList[agePreds[0].argmax()]
#         print(f'Age: {age[1:-1]} years')
#
#         cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
#                     (0, 255, 255), 2, cv2.LINE_AA)
#         cv2.imshow("Detecting age and gender", resultImg)

