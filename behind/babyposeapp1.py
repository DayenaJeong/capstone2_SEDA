import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import pickle
from collections import Counter

# 키포인트 추출 함수
def run_inference(movenet, image):
    input_image = tf.image.resize_with_pad(tf.expand_dims(image, axis=0), 192, 192)
    input_image = tf.cast(input_image, dtype=tf.int32)
    outputs = movenet(input_image)
    return outputs['output_0'].numpy()

# 비디오에서 프레임별 키포인트 추출
def extract_keypoints_from_video_frames(video_path, movenet):
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        keypoints = run_inference(movenet, frame)
        keypoints_list.append(keypoints)
    cap.release()
    return {'keypoints': keypoints_list, 'labels': []}


def calculate_mean_keypoints_from_file(keypoints_data):

    # 모든 동영상에 대한 평균 키포인트 계산
    mean_keypoints_all_videos = []
    for keypoints_list in keypoints_data['keypoints']:
        # 각 동영상에 대한 키포인트 리스트에서 평균 계산
        mean_keypoints = [[sum(pos) / len(keypoints_list) for pos in zip(*frame)] for frame in zip(*keypoints_list)]
        mean_keypoints_all_videos.append(mean_keypoints)

    return mean_keypoints_all_videos, keypoints_data['labels']

def extract_and_augment_keypoints(video_path, movenet, sampling_rate=2, additional_sampling_rate=3):
    cap = cv2.VideoCapture(video_path)
    original_keypoints = []
    flipped_keypoints = []
    sampled_keypoints = []  # sampling_rate=2에 대한 키포인트
    additional_sampled_keypoints = []  # 추가적인 sampling_rate=3에 대한 키포인트
    additional_sampled_keypoints_2 = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 원본 프레임에서 키포인트 추출
        keypoints = run_inference(movenet, frame)
        original_keypoints.append(keypoints)

        # 프레임을 좌우 반전하여 키포인트 추출
        frame_flipped = cv2.flip(frame, 1)
        keypoints_flipped = run_inference(movenet, frame_flipped)
        flipped_keypoints.append(keypoints_flipped)

        # 기존 시간적 증강: sampling_rate=2에 따라 키포인트 추출
        if frame_count % sampling_rate == 0:
            sampled_keypoints.append(keypoints)
        
        # 추가적인 시간적 증강: additional_sampling_rate=3에 따라 키포인트 추출
        if frame_count % additional_sampling_rate == 0:
            additional_sampled_keypoints.append(keypoints)
            
        # 추가적인 시간적 증강: additional_sampling_rate=3에 따라 키포인트 추출
        if frame_count % 4 == 0:
            additional_sampled_keypoints_2.append(keypoints)

        frame_count += 1
    
    cap.release()
    # 모든 키포인트 리스트와 추가적인 샘플링된 키포인트 리스트를 반환합니다.
    return original_keypoints, flipped_keypoints, sampled_keypoints, additional_sampled_keypoints, additional_sampled_keypoints_2

def calculate_keypoint_changes(keypoints_data):
    # 변경된 부분: 이미 로드된 키포인트 데이터를 직접 사용
    # 키포인트 데이터는 각 동영상의 프레임별 키포인트 리스트를 포함하는 리스트

    changes_list = []  # 변화량을 저장할 리스트 초기화

    for keypoints_list in keypoints_data['keypoints']:
        changes = []  # 개별 동영상의 키포인트 변화량을 저장할 리스트
        prev_keypoints = None

        for keypoints in keypoints_list:
            keypoints = np.array(keypoints)
            if prev_keypoints is not None:
                # 현재 프레임과 이전 프레임의 키포인트 사이의 변화량 계산
                change = np.abs(keypoints - prev_keypoints)
                changes.append(change)
            prev_keypoints = keypoints

        # 모든 변화량의 평균 계산
        if changes:
            mean_changes = np.mean(changes, axis=0)
        else:
            # 변화량이 없는 경우, 0으로 채워진 배열 반환
            mean_changes = np.zeros_like(keypoints_list[0])

        changes_list.append(mean_changes)

    return changes_list

def calculate_angle(point1, point2, point3):
    """
    세 점을 이용하여 두 벡터 사이의 각도를 계산합니다.
    :param point1, point2, point3: 각 점의 좌표를 나타내는 (x, y) 튜플이나 리스트.
    :return: 두 벡터 사이의 각도(도).
    """
    # 벡터 v1과 v2 생성
    v1 = np.array(point1) - np.array(point2)
    v2 = np.array(point3) - np.array(point2)

    # 벡터의 내적과 노름(크기)을 사용하여 각도(라디안) 계산
    angle_rad = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    # 각도를 도로 변환
    angle_deg = np.degrees(angle_rad)

    return angle_deg

# 평균
def calculate_angle_changes(keypoints_data, point_indices):
    angle_changes_list = []
    for keypoints_list in keypoints_data['keypoints']:
        angles = []
        for frame_keypoints in keypoints_list:
            # 키포인트 데이터가 충분한지 확인
            if len(frame_keypoints) > max(point_indices):
                p1 = frame_keypoints[point_indices[0]][:2]  # x, y 좌표만 사용
                p2 = frame_keypoints[point_indices[1]][:2]
                p3 = frame_keypoints[point_indices[2]][:2]
                angle = calculate_angle(p1, p2, p3)
                angles.append(angle)
            else:
                # 충분한 데이터가 없는 경우 계산에서 제외
                continue

        if angles:  # 각도 데이터가 있을 경우에만 계산
            angle_changes = np.abs(np.diff(angles))
            mean_angle_change = np.mean(angle_changes)
            angle_changes_list.append(mean_angle_change)
        else:
            # 각도 데이터가 없는 경우 0으로 처리
            angle_changes_list.append(0)

    return np.array(angle_changes_list)

# 최솟값
def calculate_min_angle_changes(keypoints_data, point_indices):
    min_angle_changes_list = []
    for keypoints_list in keypoints_data['keypoints']:
        angles = []
        for frame_keypoints in keypoints_list:
            if len(frame_keypoints) > max(point_indices):
                p1 = frame_keypoints[point_indices[0]][:2]  # x, y 좌표만 사용
                p2 = frame_keypoints[point_indices[1]][:2]
                p3 = frame_keypoints[point_indices[2]][:2]
                angle = calculate_angle(p1, p2, p3)
                angles.append(angle)

        if angles:
            angle_changes = np.abs(np.diff(angles))
            min_angle_change = np.min(angle_changes) if len(angle_changes) > 0 else 0
            min_angle_changes_list.append(min_angle_change)
        else:
            min_angle_changes_list.append(0)

    return np.array(min_angle_changes_list)

# 최댓값
def calculate_max_angle_changes(keypoints_data, point_indices):
    max_angle_changes_list = []
    for keypoints_list in keypoints_data['keypoints']:
        angles = []
        for frame_keypoints in keypoints_list:
            if len(frame_keypoints) > max(point_indices):
                p1 = frame_keypoints[point_indices[0]][:2]  # x, y 좌표만 사용
                p2 = frame_keypoints[point_indices[1]][:2]
                p3 = frame_keypoints[point_indices[2]][:2]
                angle = calculate_angle(p1, p2, p3)
                angles.append(angle)

        if angles:
            angle_changes = np.abs(np.diff(angles))
            max_angle_change = np.max(angle_changes) if len(angle_changes) > 0 else 0
            max_angle_changes_list.append(max_angle_change)
        else:
            max_angle_changes_list.append(0)

    return np.array(max_angle_changes_list)

def calculate_enhanced_autocorrelation_features(keypoints_data):
    features_list = []  # 각 동영상의 향상된 자기상관성 특성을 저장할 리스트

    for keypoints_list in keypoints_data['keypoints']:
        changes = []
        prev_keypoints = None
        for keypoints in keypoints_list:
            keypoints = np.array(keypoints)
            if prev_keypoints is not None:
                change = np.linalg.norm(keypoints - prev_keypoints)
                changes.append(change)
            prev_keypoints = keypoints

        if changes:
            changes = np.array(changes)
            autocorrelation = np.correlate(changes - np.mean(changes), changes - np.mean(changes), mode='full')
            autocorrelation = autocorrelation[autocorrelation.size // 2:]  # 자기상관성 값 중 양의 지연만 고려

            # 향상된 특성 계산
            mean_autocorrelation = np.mean(autocorrelation)
            std_autocorrelation = np.std(autocorrelation)
            peak_count = np.sum(autocorrelation > (mean_autocorrelation + std_autocorrelation))  # 평균 이상의 피크 수

            features = [mean_autocorrelation, std_autocorrelation, peak_count]
        else:
            features = [0, 0, 0]

        features_list.append(features)

    return features_list

# 추출된 키포인트로부터 특징 계산
def calculate_features(keypoints_data):
    # 키포인트에서 평균, 변화량, 자기상관성 등의 특징 계산
    mean_keypoints_all_videos, labels = calculate_mean_keypoints_from_file(keypoints_data)
    changes_list = calculate_keypoint_changes(keypoints_data)
    autocorrelation_list = calculate_enhanced_autocorrelation_features(keypoints_data)

    back_min_angle_changes_list1 = calculate_min_angle_changes(keypoints_data, (6,12,16))
    back_min_angle_changes_list2 = calculate_min_angle_changes(keypoints_data, (5,11,15))
    head_min_angle_changes_list1 = calculate_min_angle_changes(keypoints_data, (0,6,12))
    head_min_angle_changes_list2 = calculate_min_angle_changes(keypoints_data, (0,5,11))
    leg_min_angle_changes_list1 = calculate_min_angle_changes(keypoints_data, (12,14,16))
    leg_min_angle_changes_list2 = calculate_min_angle_changes(keypoints_data, (11,13,15))
    eye_min_angle_changes_list1 = calculate_min_angle_changes(keypoints_data, (1,5,9))
    eye_min_angle_changes_list2 = calculate_min_angle_changes(keypoints_data, (2,6,10))
    strech_min_angle_changes_list1 = calculate_min_angle_changes(keypoints_data, (5,8,10))
    strech_min_angle_changes_list2 = calculate_min_angle_changes(keypoints_data, (6,7,9))
    finger_min_angle_changes_list1 = calculate_min_angle_changes(keypoints_data, (0,8,10))
    finger_min_angle_changes_list2 = calculate_min_angle_changes(keypoints_data, (0,7,9))

    back_max_angle_changes_list1 = calculate_max_angle_changes(keypoints_data, (6,12,16))
    back_max_angle_changes_list2 = calculate_max_angle_changes(keypoints_data, (5,11,15))
    head_max_angle_changes_list1 = calculate_max_angle_changes(keypoints_data, (0,6,12))
    head_max_angle_changes_list2 = calculate_max_angle_changes(keypoints_data, (0,5,11))
    leg_max_angle_changes_list1 = calculate_max_angle_changes(keypoints_data, (12,14,16))
    leg_max_angle_changes_list2 = calculate_max_angle_changes(keypoints_data, (11,13,15))
    eye_max_angle_changes_list1 = calculate_max_angle_changes(keypoints_data, (1,5,9))
    eye_max_angle_changes_list2 = calculate_max_angle_changes(keypoints_data, (2,6,10))
    strech_max_angle_changes_list1 = calculate_max_angle_changes(keypoints_data, (5,8,10))
    strech_max_angle_changes_list2 = calculate_max_angle_changes(keypoints_data, (6,7,9))
    finger_max_angle_changes_list1 = calculate_max_angle_changes(keypoints_data, (0,8,10))
    finger_max_angle_changes_list2 = calculate_max_angle_changes(keypoints_data, (0,7,9))

    # 계산된 특징을 하나의 배열로 병합
    features = []
    mean_keypoints_all_videos = np.array(mean_keypoints_all_videos)
    changes_list = np.array(changes_list)
    autocorrelation_list = np.array(autocorrelation_list)
    for i in range(len(mean_keypoints_all_videos)):
        combined_feature = np.concatenate([mean_keypoints_all_videos[i].flatten(), changes_list[i].flatten(), autocorrelation_list[i].flatten(),
        # fft_features_list[i].flatten(),
        [back_max_angle_changes_list1[i] - back_min_angle_changes_list1[i],
        back_max_angle_changes_list2[i] - back_min_angle_changes_list2[i],
        head_max_angle_changes_list1[i] - head_min_angle_changes_list1[i],
        head_max_angle_changes_list2[i] - head_min_angle_changes_list2[i],
        leg_max_angle_changes_list1[i] - leg_min_angle_changes_list1[i],
        leg_max_angle_changes_list2[i] - leg_min_angle_changes_list2[i],
        eye_max_angle_changes_list1[i] - eye_min_angle_changes_list1[i],
        eye_max_angle_changes_list2[i] - eye_min_angle_changes_list2[i],
        strech_max_angle_changes_list1[i] - strech_min_angle_changes_list1[i],
        strech_max_angle_changes_list2[i] - strech_min_angle_changes_list2[i],
        finger_max_angle_changes_list1[i] - finger_min_angle_changes_list1[i],
        finger_max_angle_changes_list2[i] - finger_min_angle_changes_list2[i]]])

        features.append(combined_feature)
    return np.array(features), labels

st.title("BabyPose Estimation")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

# 레이블 매핑 딕셔너리
label_dict = {
    0: '등 젖히기',
    1: '머리 부딪히기',
    2: '다리 차기',
    3: '눈 비비기',
    4: '스트레칭',
    5: '손가락 빨기'
}

if uploaded_file is not None:
    # 파일 저장 및 Streamlit에서 비디오 재생
    video_file = uploaded_file.name
    with open(video_file, mode='wb') as f:
        f.write(uploaded_file.read())
    st.video(video_file)

    # MoveNet 모델 로드
    model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    movenet = model.signatures['serving_default']

    # 키포인트 데이터 추출
    #keypoints_data = extract_keypoints_from_video_frames(video_file, movenet)
    # 기본 키포인트 추출
    # 기본 키포인트 추출
    basic_keypoints_data = extract_keypoints_from_video_frames(video_file, movenet)

    # 키포인트 데이터 추출 및 증강
    original_keypoints, flipped_keypoints, sampled_keypoints, additional_sampled_keypoints, additional_sampled_keypoints_2 = extract_and_augment_keypoints(video_file, movenet)

    # 모든 키포인트를 하나의 리스트로 결합
    all_keypoints = basic_keypoints_data['keypoints'] + original_keypoints + flipped_keypoints + sampled_keypoints + additional_sampled_keypoints + additional_sampled_keypoints_2

    # 키포인트 데이터 딕셔너리 생성
    keypoints_data = {'keypoints': all_keypoints, 'labels': basic_keypoints_data['labels']}

    features, labels = calculate_features(keypoints_data)

    # SVM 분류기 로드 및 결과 예측
    # 모델 및 스케일러 로드
    # with open('svm_model.pkl', 'rb') as f:
    #     data = pickle.load(f)
    #     loaded_model = data['model']
    #     loaded_scaler = data['scaler']

    with open('rf_model.pkl', 'rb') as f:
        data = pickle.load(f)
        loaded_model = data['model']
        loaded_scaler = data['scaler']

    # 추출된 특징을 스케일링
    scaled_features = loaded_scaler.transform(features)

    # 결과 예측
    prediction = loaded_model.predict(scaled_features)

    # 예측 결과를 이름으로 변환
    prediction_labels = [label_dict[pred] for pred in prediction]

    # # 가장 흔한 레이블 찾기
    # most_common_label = Counter(prediction_labels).most_common(1)[0][0]

    # st.write("Classification Result:", most_common_label)

    # 레이블 카운트 계산
    label_counts = Counter(prediction_labels)
    total_predictions = sum(label_counts.values())  # 전체 레이블 수

    # 각 레이블의 출현 비율 계산
    label_percentages = {label: f"{(count / total_predictions * 100):.2f}%" for label, count in label_counts.items()}

    # 결과 출력
    st.write("Classification Results:")
    for label, percentage in label_percentages.items():
        st.write(f"{label}: {percentage}")
