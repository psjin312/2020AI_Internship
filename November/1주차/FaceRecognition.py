import dlib, cv2 # dlib : Face detection + Face recognition, cv2 : OpenCV : 이미지 작업
import numpy as np # numpy : 행렬 연산
import matplotlib.pyplot as plt # pyplot : 결과물을 그려보기 위해 사용
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

detector = dlib.get_frontal_face_detector() # 얼굴 탐지 모델
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') # 얼굴 랜드마크 탐지 모델
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat') # 얼굴 인식 모델

def find_faces(img):  # 얼굴을 찾는 함수(input을 RGB 이미지로 받음)
    dets = detector(img, 1)  # 찾은 얼굴의 결과물

    if len(dets) == 0:  # 찾은 얼굴이 없다면
        return np.empty(0), np.empty(0), np.empty(0)  # 빈 배열들 반환, 로직 끝
    
    rects, shapes = [], []
    shapes_np = np.zeros((len(dets), 68, 2), dtype=np.int)  
    # shape : 얼굴의 랜드마크, 68개의 점을 구하는 함수 만들기
    for k, d in enumerate(dets):  # 찾은 얼굴 개수만큼 loop 돌기
        rect = ((d.left(), d.top()), (d.right(), d.bottom()))  # rect 변수에 얼굴(사각형 박스)의 왼쪽, 위, 오른쪽, 아래 좌표 넣기
        rects.append(rect)  # rects에 차곡차곡 쌓기

        shape = sp(img, d)  # 이미지와 사각형을 넣으면 68개의 점이 나옴
        
        for i in range(0, 68):  # dlib shape를 numpy array로 변환
            shapes_np[k][i] = (shape.part(i).x, shape.part(i).y)

        shapes.append(shape)  # 랜드마크 결과물을 차곡차곡 쌓기
        
    return rects, shapes, shapes_np  # 리턴해주면 얼굴 찾는 함수 끝

def encode_faces(img, shapes):  # 얼굴 인코드하는 함수
# 얼굴을 인코드한다 : 사람 얼굴 이미지에서 랜드마크들이 점으로 있다, 눈, 코, 눈썹, 입 턱 등등의 랜드마크 정보를
# Encoder에 넣으면 128개의 벡터가 나옴. 벡터의 특징(숫자)들로 사람의 얼굴 구분
# 128개의 벡터의 거리가 얼마나 멀고 가깝냐에 따라 같은 사람인지, 아닌 사람인지 구분

    face_descriptors = []  # 결과값을 저장할 리스트
    for shape in shapes:  
    # 위에서 구한 랜드마크들의 배열 집합 크기만큼 loop 돌면서 face recognition 모델 돌림
    # compute_face_descriptor라는 메소드 사용
        face_descriptor = facerec.compute_face_descriptor(img, shape) # 이미지와 랜드마크 사용
        face_descriptors.append(np.array(face_descriptor))  # 결과값을 numpyarray로 바꿔서 차곡차곡 쌓기

    return np.array(face_descriptors) # 그 값 반환

# 미리 저장해놓은 사용자들의 얼굴에 인코드된 데이터를 미리 저장 
img_paths = {
    'songkangho': 'img/parasite/songkangho1.jpg',
    'parksodam': 'img/parasite/parksodam1.jpg',
    'choiwoosik': 'img/parasite/choiwoosik1.jpg'
}

descs = {  # 계산한 결과를 저장할 변수
    'songkangho': None,
    'parksodam': None,
    'choiwoosik': None
}

for name, img_path in img_paths.items():  
# 이미지 path만큼 loop 돌면서 openCV의 cv2.imread로 이미지를 읽음(이미지 로드)
    img_bgr = cv2.imread(img_path)  #  BGR 형식으로 나옴
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # cv2.cvtColor() : 컬러 체계를 바꿈 => BGR을 RGB로 바꿈

    _, img_shapes, _ = find_faces(img_rgb) # RGB로 바꾼 이미지에서 얼굴을 찾아서 shape들을 받아옴(랜드마크)
    descs[name] = encode_faces(img_rgb, img_shapes)[0] 
    # encode_faces 함수에 전체 이미지와 각 사람의 랜드마크를 넣어줌
    # 인코딩된 결과를 각 사람의 이름에 맞게 저장

np.save('img/descs.npy', descs) # 결과값을 np.save 함수를 통해 .npy 함수로 써줌
print(descs) # descs 확인



img_bgr = cv2.imread('img/parasite/poster1.jpg')  # poster1.jpg 파일 읽기 
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # 이미지 컬러 RGB로 변환

rects, shapes, _ = find_faces(img_rgb)
descriptors = encode_faces(img_rgb, shapes) # 얼굴을 인코드한 결과를 descriptors로 받아옴


# 결과값 뿌려주기
fig, ax = plt.subplots(1, figsize=(20, 20))
ax.imshow(img_rgb)

for i, desc in enumerate(descriptors):  # descriptors만큼 loop 돌기
    
    found = False
    for name, saved_desc in descs.items(): # 얼굴들의 인코드된 값을 저장해놓은 descs
        dist = np.linalg.norm([desc] - saved_desc, axis=1)
        # distance, linearalgebra norm 함수
        # a, b 벡터 사이의 유클리드 거리(유클리디안 distance)를 구함(np.linalg.norm(a-b))
        # 2차원 그래프상에서 각 사람의 좌표값에 얼마나 가까운지를 통해 그 사람임을 판단

        if dist < 0.48:  
        # 0.6으로 했을 때 성능이 가장 좋다고 함
        # 얼굴의 특징과 서로 차이가 뚜렷하지 않을 때에는 값을 낮추는 것이 정확도가 높음을 확인
            found = True

            text = ax.text(rects[i][0][0], rects[i][0][1], name,  # 찾게 되면 그 사람의 name을 쓰기
                    color='b', fontsize=40, fontweight='bold')
            text.set_path_effects([path_effects.Stroke(linewidth=10, foreground='white'), path_effects.Normal()])
            rect = patches.Rectangle(rects[i][0],  # 얼굴 부분에 사각형 그리기
                                 rects[i][1][1] - rects[i][0][1],
                                 rects[i][1][0] - rects[i][0][0],
                                 linewidth=2, edgecolor='w', facecolor='none')
            ax.add_patch(rect)

            break
    
    if not found:  # 얼굴은 찾았는데 누군지 모를 때 unknown이라고 표시
        ax.text(rects[i][0][0], rects[i][0][1], 'unknown',
                color='r', fontsize=20, fontweight='bold')
        rect = patches.Rectangle(rects[i][0],
                             rects[i][1][1] - rects[i][0][1],
                             rects[i][1][0] - rects[i][0][0],
                             linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

plt.axis('off')
plt.savefig('result/output.png')
plt.show()
