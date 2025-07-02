import onnxruntime as ort
import cv2
import argparse
import math
import copy
from shapely.geometry import Polygon
import pyclipper
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def resize_img(img, input_size=600):
    """
    resize img and limit the longest side of the image to input_size
    """
    img = np.array(img)
    im_shape = img.shape
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(input_size) / float(im_size_max)
    img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale)
    return img

def str_count(s):
    """
    Count the number of Chinese characters,
    a single English character and a single number
    equal to half the length of Chinese characters.
    args:
        s(string): the input of string
    return(int):
        the number of Chinese characters
    """
    import string

    count_zh = count_pu = 0
    s_len = len(str(s))
    en_dg_count = 0
    for c in str(s):
        if c in string.ascii_letters or c.isdigit() or c.isspace():
            en_dg_count += 1
        elif c.isalpha():
            count_zh += 1
        else:
            count_pu += 1
    return s_len - math.ceil(en_dg_count / 2)

def text_visual(
    texts,
    scores,
    img_h=400,
    img_w=600,
    threshold=0.0,
    font_path=str("simfang.ttf"),
):
    """
    create new blank img and draw txt on it
    args:
        texts(list): the text will be draw
        scores(list|None): corresponding score of each txt
        img_h(int): the height of blank img
        img_w(int): the width of blank img
        font_path: the path of font which is used to draw text
    return(array):
    """
    if scores is not None:
        assert len(texts) == len(
            scores
        ), "The number of txts and corresponding scores must match"

    def create_blank_img():
        blank_img = np.ones(shape=[img_h, img_w], dtype=np.uint8) * 255
        blank_img[:, img_w - 1 :] = 0
        blank_img = Image.fromarray(blank_img).convert("RGB")
        draw_txt = ImageDraw.Draw(blank_img)
        return blank_img, draw_txt

    blank_img, draw_txt = create_blank_img()

    font_size = 20
    txt_color = (0, 0, 0)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

    gap = font_size + 5
    txt_img_list = []
    count, index = 1, 0
    for idx, txt in enumerate(texts):
        index += 1
        if scores[idx] < threshold or math.isnan(scores[idx]):
            index -= 1
            continue
        first_line = True
        while str_count(txt) >= img_w // font_size - 4:
            tmp = txt
            txt = tmp[: img_w // font_size - 4]
            if first_line:
                new_txt = str(index) + ": " + txt
                first_line = False
            else:
                new_txt = "    " + txt
            draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
            txt = tmp[img_w // font_size - 4 :]
            if count >= img_h // gap - 1:
                txt_img_list.append(np.array(blank_img))
                blank_img, draw_txt = create_blank_img()
                count = 0
            count += 1
        if first_line:
            new_txt = str(index) + ": " + txt + "   " + "%.3f" % (scores[idx])
        else:
            new_txt = "  " + txt + "  " + "%.3f" % (scores[idx])
        draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
        # whether add new blank img or not
        if count >= img_h // gap - 1 and idx + 1 < len(texts):
            txt_img_list.append(np.array(blank_img))
            blank_img, draw_txt = create_blank_img()
            count = 0
        count += 1
    txt_img_list.append(np.array(blank_img))
    if len(txt_img_list) == 1:
        blank_img = np.array(txt_img_list[0])
    else:
        blank_img = np.concatenate(txt_img_list, axis=1)
    return np.array(blank_img)

def draw_ocr(
    image,
    boxes,
    txts=None,
    scores=None,
    drop_score=0.5,
    font_path=str("simfang.ttf"),
):
    """
    Visualize the results of OCR detection and recognition
    args:
        image(Image|array): RGB image
        boxes(list): boxes with shape(N, 4, 2)
        txts(list): the texts
        scores(list): txxs corresponding scores
        drop_score(float): only scores greater than drop_threshold will be visualized
        font_path: the path of font which is used to draw text
    return(array):
        the visualized img
    """
    if scores is None:
        scores = [1] * len(boxes)
    box_num = len(boxes)
    for i in range(box_num):
        if scores is not None and (scores[i] < drop_score or math.isnan(scores[i])):
            continue
        box = np.reshape(np.array(boxes[i]), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
    if txts is not None:
        img = np.array(resize_img(image, input_size=600))
        txt_img = text_visual(
            txts,
            scores,
            img_h=img.shape[0],
            img_w=600,
            threshold=drop_score,
            font_path=font_path,
        )
        img = np.concatenate([np.array(img), np.array(txt_img)], axis=1)
        return img
    return image

def sav2Img(org_img, result, name="draw_ocr.jpg"):
    result = result[0]
    image = org_img[:, :, ::-1]
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores)
    im_show = Image.fromarray(im_show)
    im_show.save(name)

def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [
        points[index_1], points[index_2], points[index_3], points[index_4]
    ]
    return box, min(bounding_box[1])

def box_score_fast(bitmap, _box):
    '''
    box_score_fast: use bbox mean score as the mean score
    '''
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

def unclip(box, unclip_ratio):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded
    
def boxes_from_bitmap(pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        '''
        box_thresh = 0.6
        max_candidates = 1000
        unclip_ratio = 1.5
        min_size = 3

        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = get_mini_boxes(contour)
            if sside < min_size:
                continue
            points = np.array(points)
            score = box_score_fast(pred, points.reshape(-1, 2))
            if box_thresh > score:
                continue

            box = unclip(points, unclip_ratio).reshape(-1, 1, 2)
            box, sside = get_mini_boxes(box)
            if sside < min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype("int32"))
            scores.append(score)
        return np.array(boxes, dtype="int32"), scores

def order_points_clockwise(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
    diff = np.diff(np.array(tmp), axis=1)
    rect[1] = tmp[np.argmin(diff)]
    rect[3] = tmp[np.argmax(diff)]
    return rect

def clip_det_res(points, img_height, img_width):
    for pno in range(points.shape[0]):
        points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
        points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
    return points
    
def filter_tag_det_res(dt_boxes, image_shape):
    img_height, img_width = image_shape[0:2]
    dt_boxes_new = []
    for box in dt_boxes:
        if type(box) is list:
            box = np.array(box)
        box = order_points_clockwise(box)
        box = clip_det_res(box, img_height, img_width)
        rect_width = int(np.linalg.norm(box[0] - box[1]))
        rect_height = int(np.linalg.norm(box[0] - box[3]))
        if rect_width <= 3 or rect_height <= 3:
            continue
        dt_boxes_new.append(box)
    dt_boxes = np.array(dt_boxes_new)
    return dt_boxes

def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and (
                _boxes[j + 1][0][0] < _boxes[j][0][0]
            ):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes

def get_rotate_crop_image(img, points):
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])
        )
    )
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])
        )
    )
    pts_std = np.float32(
        [
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height],
        ]
    )
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M,
        (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img

def resize_norm_img(img,shape):
    h, w = img.shape[:2]
    imgC,imgH,imgW = shape
    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
        
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype("float32")
    resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im

def decode(dict_character,text_index, text_prob=None, is_remove_duplicate=False):
    """convert text-index into text-label."""
    result_list = []
    ignored_tokens = [0]
    batch_size = len(text_index)

    for batch_idx in range(batch_size):
        selection = np.ones(len(text_index[batch_idx]), dtype=bool)
        if is_remove_duplicate:
            selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]
        for ignored_token in ignored_tokens:
            selection &= text_index[batch_idx] != ignored_token

        char_list = [
            dict_character[text_id] for text_id in text_index[batch_idx][selection]
        ]
        conf_list = text_prob[batch_idx][selection]
        if len(conf_list) == 0:
            conf_list = [0]

        text = "".join(char_list)
        result_list.append((text, np.mean(conf_list).tolist()))
    return result_list

def det_postprocess(outs_dict, shape_list):
        pred = outs_dict['maps']
        pred = pred[:, 0, :, :]
        segmentation = pred > 0.3
        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            mask = segmentation[batch_index]
            boxes, scores = boxes_from_bitmap(pred[batch_index], mask, src_w, src_h)
            boxes_batch.append({'points': boxes})
        return boxes_batch

def cls_postprocess(preds,label_list):
    pred_idxs = preds.argmax(axis=1)
    decode_out = [(label_list[idx], preds[i, idx])
                    for i, idx in enumerate(pred_idxs)]
    return decode_out
    
def rec_postprocess(preds,character_dict_path,use_space_char):
    character_str=[]
    with open(character_dict_path, "rb") as fin:
        lines = fin.readlines()
        for line in lines:
            line = line.decode("utf-8").strip("\n").strip("\r\n")
            character_str.append(line)
    if use_space_char:
        character_str.append(" ")
    dict_character = list(character_str)
    dict_character = ["blank"] + dict_character
    if isinstance(preds, tuple) or isinstance(preds, list):
        preds = preds[-1]
    preds_idx = preds.argmax(axis=2)
    preds_prob = preds.max(axis=2)
    text = decode(dict_character,preds_idx, preds_prob, is_remove_duplicate=True)

    return text

def text_detector(session,img,shape=[960,960]):
    orig_h, orig_w = img.shape[:2]
    image = cv2.resize(img, shape)
    mean = np.array([123.675, 116.28, 103.53],dtype=np.float32).reshape(1,1,3)
    std = np.array([58.395, 57.12, 57.375],dtype=np.float32).reshape(1,1,3)
    image = (image-mean)/std 
    image = image.transpose(2,0,1)
    image = np.expand_dims(image, axis=0).astype(np.float32)
    shape_list = [[orig_h, orig_w, shape[1]/orig_h, shape[0]/orig_w]]

    det_out = session.run(None,input_feed={'x':image})
    det_preds = {}
    det_preds["maps"] = det_out[0]

    post_result = det_postprocess(det_preds, shape_list)
    dt_boxes = post_result[0]["points"]
    dt_boxes = filter_tag_det_res(dt_boxes, img.shape)
    if dt_boxes is None:
        return None, None
    dt_boxes = sorted_boxes(dt_boxes)

    return dt_boxes

def text_classifier(session,img_list,shape=[3,80,160]):
    img_list = copy.deepcopy(img_list)
    img_num = len(img_list)
    # Calculate the aspect ratio of all text bars
    width_list = []
    for img in img_list:
        width_list.append(img.shape[1] / float(img.shape[0]))
    # Sorting can speed up the cls process
    indices = np.argsort(np.array(width_list))

    cls_res = [["", 0.0]] * img_num
    batch_num = 1

    for beg_img_no in range(0, img_num, batch_num):

        end_img_no = min(img_num, beg_img_no + batch_num)
        norm_img_batch = []
        for ino in range(beg_img_no, end_img_no):
            norm_img = resize_norm_img(img_list[indices[ino]],shape)
            norm_img = norm_img[np.newaxis, :]
            norm_img_batch.append(norm_img)
        norm_img_batch = np.concatenate(norm_img_batch)
        norm_img_batch = norm_img_batch.copy()

        outputs = session.run(None,input_feed={'x':norm_img_batch})
        prob_out = outputs[0]
        cls_result = cls_postprocess(prob_out,label_list=["0", "180"])

        for rno in range(len(cls_result)):
            label, score = cls_result[rno]
            cls_res[indices[beg_img_no + rno]] = [label, score]
            if "180" in label and score > 0.9:
                img_list[indices[beg_img_no + rno]] = cv2.rotate(
                    img_list[indices[beg_img_no + rno]], 1
                )
    return img_list, cls_res

def text_recognizer(session,img_list,shape=[3,48,320],character_dict_path=r"./ppocrv5_dict.txt"):
    img_num = len(img_list)
    # Calculate the aspect ratio of all text bars
    width_list = []
    for img in img_list:
        width_list.append(img.shape[1] / float(img.shape[0]))
    # Sorting can speed up the recognition process
    indices = np.argsort(np.array(width_list))
    rec_res = [["", 0.0]] * img_num
    batch_num = 1

    for beg_img_no in range(0, img_num, batch_num):
        end_img_no = min(img_num, beg_img_no + batch_num)
        norm_img_batch = []
        for ino in range(beg_img_no, end_img_no):
            norm_img = resize_norm_img(img_list[indices[ino]],shape)
            norm_img = norm_img[np.newaxis, :]
            norm_img_batch.append(norm_img)

        norm_img_batch = np.concatenate(norm_img_batch)
        norm_img_batch = norm_img_batch.copy()

        outputs = session.run(None,input_feed={'x':norm_img_batch})
        preds = outputs[0]
        rec_result = rec_postprocess(preds,character_dict_path,use_space_char=True)
        for rno in range(len(rec_result)):
            rec_res[indices[beg_img_no + rno]] = rec_result[rno]

    return rec_res


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path",type=str,default=str(r"./test_pic/11.jpg"),help="Path to input image.")
    parser.add_argument("--det_model_dir",type=str,default=str(r"../model/onnx/det_mobile_sim_static.onnx"),help="Path to detection model.")
    parser.add_argument("--rec_model_dir",type=str,default=str(r"../model/onnx/rec_mobile_sim_static.onnx"),help="Path to recognition model.")
    parser.add_argument("--cls_model_dir",type=str,default=str(r"../model/onnx/cls_mobile_sim_static.onnx"),help="Path to classification model.")
    parser.add_argument("--character_dict_path",type=str,default=str(r"./ppocrv5_dict.txt"),help="recognition dictionary")
    parser.add_argument("--det_limit_side_len", type=float, default=[960,960],help="detection model input size")
    parser.add_argument("--rec_image_shape", type=str, default=[3, 48, 320],help="recognition model input size")
    parser.add_argument("--cls_image_shape", type=str, default=[3, 80, 160],help="classification model input size")
    
    return parser.parse_args()

def main(args):
    det_session = ort.InferenceSession(args.det_model_dir,providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    rec_session = ort.InferenceSession(args.rec_model_dir,providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    cls_session = ort.InferenceSession(args.cls_model_dir,providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    image=cv2.imread(args.img_path)

    #文字检测
    dt_boxes = text_detector(det_session,image,args.det_limit_side_len)

    # 图片裁剪
    img_crop_list = []
    im = image.copy()
    for bno in range(len(dt_boxes)):
        tmp_box = copy.deepcopy(dt_boxes[bno])
        img_crop = get_rotate_crop_image(im, tmp_box)
        img_crop_list.append(img_crop)
        # cv2.imwrite(f"{bno}.jpg",img_crop)

    # 方向分类
    img_crop_list, angle_list = text_classifier(cls_session,img_crop_list,args.cls_image_shape)

    # 文字识别
    rec_res = text_recognizer(rec_session,img_crop_list,args.rec_image_shape,args.character_dict_path)
    filter_boxes, filter_rec_res = [], []
    for box, rec_result in zip(dt_boxes, rec_res):
        text, score = rec_result
        if score >= 0.5:
            filter_boxes.append(box)
            filter_rec_res.append(rec_result)

    #输出结果
    ocr_res=[]
    tmp_res = [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]
    ocr_res.append(tmp_res)
    for box in ocr_res[0]:
        print(box)
    sav2Img(image, ocr_res,name='res_onnx.jpg')

if __name__=='__main__':
    args=init_args()
    main(args)