"""
特徴抽出器であるConvNeXtの性能を測るためのコード
〜実行コマンド例（BG_335の時）〜
python test_convnext.py BG_335
"""
import cv2, skimage
import numpy as np
import sys, torch
import torch.nn as nn
from model_extract_features import ConvNextEncoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = sys.argv
video_name = args[1]
counted_keyframes = 0

# 前処理
# 呼び出し元：preprocess_frame関数
def resize_frame(image, target_height=224, target_width=224):
    if len(image.shape) == 2:
        # シングルチャンネルのグレースケール画像を3回コピーして3チャンネルの画像にする．
        image = np.tile(image[:, :, None], 3)
    elif len(image.shape) == 4:
        image = image[:, :, :, 0]

    height, width, channels = image.shape
    if height == width:
        resized_image = cv2.resize(image, (target_height, target_width))
    elif height < width:
        resized_image = cv2.resize(image, (int(width * target_height / height),
                                           target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,
                                      cropping_length:resized_image.shape[1] - cropping_length]
    else:
        resized_image = cv2.resize(image, (target_height,
                                           int(height * target_width / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:
                                      resized_image.shape[0] - cropping_length]
    return cv2.resize(resized_image, (target_height, target_width))


# 前処理
# 呼び出し元：sample_frames関数
def preprocess_frame(image, target_height=224, target_width=224):
    image = resize_frame(image, target_height, target_width)
    image = skimage.img_as_float(image).astype(np.float32)
    # ILSVRCデータセットの画像の平均（RGB形式）に基づくホワイトニング
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    return image


def extract_features():
  path_keyframes_txt = "/home/kouki/Codex/all_frames/" + video_name +"_keyframes.txt"
  with open(path_keyframes_txt) as f:
      lines = f.readlines()
  counted_keyframes = len(lines)

  encoder = ConvNextEncoder()
  for i in range(counted_keyframes):
      key_frame_num = int(lines[i])
      print("frame :{}".format(key_frame_num))

      image = cv2.imread('/home/kouki/Codex/all_frames/' + video_name + '/' + str(i) +
                         '-' + str(key_frame_num) + '.png')
      # 前処理
      image = preprocess_frame(image)
      # 画像を2048次元の特徴量に変換
      features2048 = encoder.encode(image)  # type: torch.FloatTensor
      if i == 0:
          features_array = features2048
      else:
          features_array = torch.cat([features_array, features2048], dim=0)

  return features_array


def main():
    features_array = extract_features()
    cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
    input1 = features_array[0]
    max_cosine_output = -2
    similar_frame_num = -1
    for i in range(counted_keyframes):
        if i == 0:
            continue
        print(i)
        input2 = features_array[i]
        cosine_output = cosine_similarity(input1, input2)
        if cosine_output > max_cosine_output:
            max_cosine_output = cosine_output
            similar_frame_num = i

    print(similar_frame_num)


if __name__ == "__main__":
    main()