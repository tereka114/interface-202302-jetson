
import sys
import argparse
import numpy as np
import cv2
import torch
import time

from yolox.exp import get_exp
from yolox.utils import postprocess
from yolox.data.data_augment import ValTransform
from yolox.utils.visualize import vis
from yolox.data.datasets import COCO_CLASSES


def load_model(exp, checkpoint_path, trt_file):
    """
    checkpoint_path: モデル保存先のパス
    """
    exp = get_exp(exp)
    model = exp.get_model()
    model.cuda()
    model.eval()
    if trt_file is None:
        # get custom trained checkpoint
        ckpt_file = checkpoint_path
        ckpt = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(ckpt["model"])
    else:
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            model(x)
            model = model_trt
    return model

def yolox_inference(img, model, test_size, num_classes, confthre, nmsthre): 
    bboxes = []
    bbclasses = []
    scores = []
    
    preproc = ValTransform(legacy = False)

    tensor_img, _ = preproc(img, None, test_size)
    tensor_img = torch.from_numpy(tensor_img).unsqueeze(0)
    tensor_img = tensor_img.float()
    tensor_img = tensor_img.cuda()

    with torch.no_grad():
        outputs = model(tensor_img)
        outputs = postprocess(
                    outputs, num_classes, confthre,
                    nmsthre, class_agnostic=True
                )

    if outputs[0] is None:
        return [], [], []
    
    outputs = outputs[0].cpu()
    bboxes = outputs[:, 0:4]

    bboxes /= min(test_size[0] / img.shape[0], test_size[1] / img.shape[1])
    bbclasses = outputs[:, 6]
    scores = outputs[:, 4] * outputs[:, 5]
    
    return bboxes, bbclasses, scores

def draw_image(img, bboxes, scores, bbclasses):
    """
    img: 画像
    scores
    """
    for bbox, score, bclass in zip(bboxes, scores, bbclasses):
        # 取得してきたBoundingBoxの中で、人間と予測したもののみ動かす。
        if bclass == 0:
            img = cv2.rectangle(
                img=img.astype(np.uint8), 
                pt1=(int(bbox[0]), int(bbox[1])), 
                pt2=(int(bbox[2]), int(bbox[3])), 
                color=(255,0,0), 
                thickness=4
            )
    return img
              

def define_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp", help="exp file or model name")
    parser.add_argument("-m", "--model", help="model file")
    parser.add_argument("--trt_file", default=None, help="model file")
    parser.add_argument("--speed_test",default=False,action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    # Step0:準備
    args = define_argument()

    # Step1:モデルの読み込み
    model = load_model(args.exp, args.model, args.trt_file)

    # Step2:キャプチャ取り込みの準備
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print ("cannot open Video Stream")
        sys.exit()

    test_size = (640, 640)
    # 無限ループ
    step = 0

    model_inference_times = []
    process_inference_times = []
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        # Step3:フレームごとに推論をする
        frame = frame[:,:,::-1]
        start_inference_time = time.time()
        bboxes, bbclasses, scores = yolox_inference(frame, model, test_size, 80, 0.45, 0.6)
        end_inference_time = time.time()
        # Step4:結果を描画する。
        vis_frame = draw_image(frame, bboxes, scores, bbclasses)

        # Step5:結果をウィンドウに表示する。
        if args.speed_test is False:
            cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
            cv2.imshow("yolox", vis_frame[:, :, ::-1])
        end_time = time.time()
        print ("Inference Time :", end_time - start_time, "Model Inference Time:",end_inference_time - start_inference_time)
        
        model_inference_times.append(end_inference_time - start_inference_time)
        process_inference_times.append(end_time - start_time)

        # キャンセルボタンをの入力を待つ。
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break
        step += 1
        if step == 1000 and args.speed_test:
            break
        # cv2.imwrite(f"outputs/img_{step}.png", vis_frame[:, :, ::-1])
    print ("Avg Inference Time :", np.mean(process_inference_times), "Avg Model Inference Time:",np.mean(model_inference_times))
