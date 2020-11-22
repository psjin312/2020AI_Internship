# 마스크 착용 확률 판단하기

요즘은 코로나 감염, 전염을 막기 위해 마스크를 필수적으로 착용해야 한다. 이에 마스크 착용을 인식하고 확률을 판단하는 실습을 진행해보았다.

MobileNetV2를 사용해 Transfer learning
```python
# 사진에서 마스크 인식하여 확률 나타내기

# tensorflow 2.x 버전 이상
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

plt.style.use('dark_background')

# Load Models
facenet = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel') # Face Detection 모델, opencv dnn 모델 사용해서 load
model = load_model('models/mask_detector.model') # Mask Detector, keras model

# Load Image
img = cv2.imread('imgs/01.jpg') # test image load
h, w = img.shape[:2]

plt.figure(figsize=(16, 10))
plt.imshow(img[:, :, ::-1]) # opencv를 읽으면 BGR로 읽히기 때문에 RGB로 채널 변환

# Preprocess Image for Face Detection
blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(300, 300), mean=(104., 177., 123.)) # cv2.dnn.blobFromImage() : dnn 모듈이 사용하는 형태로 이미지를 변형, axis 순서만 바뀜
facenet.setInput(blob) # 모델에 input 데이터를 넣어줌
dets = facenet.forward() # 결과 추론

# Detect Faces
faces = []

for i in range(dets.shape[2]): # detection 한 결과가 얼마나 자신있는지
    confidence = dets[0, 0, i, 2]
    
    if confidence < 0.5: # 0.5 미만이면 넘기기
        continue

    x1 = int(dets[0, 0, i, 3] * w) # x, y 바운딩 박스 구함
    y1 = int(dets[0, 0, i, 4] * h)
    x2 = int(dets[0, 0, i, 5] * w)
    y2 = int(dets[0, 0, i, 6] * h)
    
    face = img[y1:y2, x1:x2] # 원본 이미지에서 얼굴만 잘라내서 저장
    faces.append(face)

plt.figure(figsize=(16, 5))

for i, face in enumerate(faces):
    plt.subplot(1, len(faces), i+1)
    plt.imshow(face[:, :, ::-1])
    
# Detect Masks from Faces
plt.figure(figsize=(16, 5))

for i, face in enumerate(faces):
    face_input = cv2.resize(face, dsize=(224, 224))
    face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
    face_input = preprocess_input(face_input) # mobileNetV2에서 preprocessing 하는것과 똑같이 해주기 위함
    face_input = np.expand_dims(face_input, axis=0) 
    # 원래는 (224, 224, 3)이 나와야 하는데 (1, 224,224, 3)으로 넣기 위해 np.expand_dims()를 이용해 0번 axis에 차원을 추가함
    
    mask, nomask = model.predict(face_input).squeeze()
    # load 해놓은 모델에 predict() 함수로 output (마스크 쓴 확률, 안 쓴 확률)

    plt.subplot(1, len(faces), i+1)
    plt.imshow(face[:, :, ::-1])
    
    
    # plt.savefig('output.png')
    
    
    
    plt.title('%.2f%%' % (mask * 100)) # 마스크 썼는지 확률

```


Cross-Entropy를 이용해 학습
---
![image](https://user-images.githubusercontent.com/34376342/99876471-33941480-2c3a-11eb-808c-c6e09ab75304.png)

크로스 엔트로피란?
실제 분포  *q*에 대해 알지 못하는 상태에서, 모델링을 통해 구한 분포인  *p*를 통하여  *q*를 예측하는 것이다.  **즉, *q*와  *p*가 모두 들어가서 크로스 엔트로피**라고 한다.

머신러닝을 하는 경우에 실제 환경의 값과  *q*, 예측값(관찰값) *p*를 모두 알고 있는 경우가 있다.  **즉, 머신러닝의 모델은 몇%의 확률로 예측했는데, 실제 확률은 몇%이다.** 라는 사실을 알고 있을 때 사용한다.

크로스 엔트로피에서는 실제값과 예측값이 맞는 경우에는 0으로 수렴하고, 값이 틀릴 경우에는 값이 커지기 때문에,  `실제 값과 예측 값의 차이를 줄이기 위한 엔트로피`라고 보면 된다.

사진에서 마스크 착용률 판단하기
---
```python
# 사진에서 마스크 인식하여 확률 나타내기

# tensorflow 2.x 버전 이상
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

plt.style.use('dark_background')

# Load Models
facenet = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel') # Face Detection 모델, opencv dnn 모델 사용해서 load
model = load_model('models/mask_detector.model') # Mask Detector, keras model

# Load Image
img = cv2.imread('imgs/01.jpg') # test image load
h, w = img.shape[:2]

plt.figure(figsize=(16, 10))
plt.imshow(img[:, :, ::-1]) # opencv를 읽으면 BGR로 읽히기 때문에 RGB로 채널 변환

# Preprocess Image for Face Detection
blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(300, 300), mean=(104., 177., 123.)) # cv2.dnn.blobFromImage() : dnn 모듈이 사용하는 형태로 이미지를 변형, axis 순서만 바뀜
facenet.setInput(blob) # 모델에 input 데이터를 넣어줌
dets = facenet.forward() # 결과 추론

# Detect Faces
faces = []

for i in range(dets.shape[2]): # detection 한 결과가 얼마나 자신있는지
    confidence = dets[0, 0, i, 2]
    
    if confidence < 0.5: # 0.5 미만이면 넘기기
        continue

    x1 = int(dets[0, 0, i, 3] * w) # x, y 바운딩 박스 구함
    y1 = int(dets[0, 0, i, 4] * h)
    x2 = int(dets[0, 0, i, 5] * w)
    y2 = int(dets[0, 0, i, 6] * h)
    
    face = img[y1:y2, x1:x2] # 원본 이미지에서 얼굴만 잘라내서 저장
    faces.append(face)

plt.figure(figsize=(16, 5))

for i, face in enumerate(faces):
    plt.subplot(1, len(faces), i+1)
    plt.imshow(face[:, :, ::-1])
    
# Detect Masks from Faces
plt.figure(figsize=(16, 5))

for i, face in enumerate(faces):
    face_input = cv2.resize(face, dsize=(224, 224))
    face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
    face_input = preprocess_input(face_input) # mobileNetV2에서 preprocessing 하는것과 똑같이 해주기 위함
    face_input = np.expand_dims(face_input, axis=0) 
    # 원래는 (224, 224, 3)이 나와야 하는데 (1, 224,224, 3)으로 넣기 위해 np.expand_dims()를 이용해 0번 axis에 차원을 추가함
    
    mask, nomask = model.predict(face_input).squeeze()
    # load 해놓은 모델에 predict() 함수로 output (마스크 쓴 확률, 안 쓴 확률)

    plt.subplot(1, len(faces), i+1)
    plt.imshow(face[:, :, ::-1])
    
    # plt.savefig('output.png')
    
    plt.title('%.2f%%' % (mask * 100)) # 마스크 썼는지 확률
```
동영상에서 마스크 착용률 판단하기
---
```python
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2

facenet = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel') # Face Detection 모델, opencv dnn 모델 사용해서 load
model = load_model('models/mask_detector.model') # Mask Detector, keras model

# cap = cv2.VideoCapture('imgs/01.mp4')
cap = cv2.VideoCapture('imgs/Test2.mp4') # 비디오 입력
ret, img = cap.read()

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# out = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (img.shape[1], img.shape[0]))
out = cv2.VideoWriter('TestOutput.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (img.shape[1], img.shape[0]))


while cap.isOpened():
    ret, img = cap.read() # 비디오 프레임을 제대로 읽었다면 ret 값이 True, 실패하면 False, 읽은 프레임 : img
    if not ret:
        break

    h, w = img.shape[:2] # 이미지의 높이와 너비 추출

    # Preprocess Image for Face Detection
    blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(300, 300), mean=(104., 177., 123.))
    facenet.setInput(blob) # 모델에 input 데이터를 넣어줌
    dets = facenet.forward() # 결과 추론, 얼굴 추출 결과를 dets에 저장

    result_img = img.copy()

    for i in range(dets.shape[2]): # detection 한 결과가 얼마나 자신있는지(신뢰도)
        confidence = dets[0, 0, i, 2]
        if confidence < 0.5:  
            continue

        x1 = int(dets[0, 0, i, 3] * w) # x, y 바운딩 박스 구함
        y1 = int(dets[0, 0, i, 4] * h)
        x2 = int(dets[0, 0, i, 5] * w)
        y2 = int(dets[0, 0, i, 6] * h)
        
        face = img[y1:y2, x1:x2] # 원본 이미지에서 얼굴 영역 추출

        # 추출한 얼굴 영역 preprocess
        face_input = cv2.resize(face, dsize=(224, 224))
        face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
        face_input = preprocess_input(face_input) # mobileNetV2에서 preprocessing 하는것과 똑같이 해주기 위함
        face_input = np.expand_dims(face_input, axis=0)
        # 원래는 (224, 224, 3)이 나와야 하는데 (1, 224,224, 3)으로 넣기 위해 np.expand_dims()를 이용해 0번 axis에 차원을 추가함

        
        mask, nomask = model.predict(face_input).squeeze()
        # 마스크 검출 모델로 결과값 return. load 해놓은 모델에 predict() 함수로 output (마스크 쓴 확률, 안 쓴 확률)

        # 라벨링
        if mask > nomask: # 마스크 쓴 확률이 높다면
            color = (0, 255, 0)
            label = 'Mask %d%%' % (mask * 100) # 마스크 쓴 확률 표시
        else:
            color = (0, 0, 255)
            label = 'No Mask %d%%' % (nomask * 100) # 마스크 안 쓴 확률 표시

        # 화면에 얼굴 영역 표시 및 마스크 착용 여부 출력
        cv2.rectangle(result_img, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA)
        cv2.putText(result_img, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color, thickness=2, lineType=cv2.LINE_AA)

    out.write(result_img)
    cv2.imshow('result', result_img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): # q를 눌러 종료
        break

out.release()
cap.release()
```

Reference
---
**DataSet(with mask, without mask)** :  [https://github.com/prajnasb/observations](https://github.com/prajnasb/observations)
