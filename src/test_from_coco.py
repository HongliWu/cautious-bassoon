from email.mime import image
import pickle
import json
import numpy as np
import metric
import cv2


class TestCOCO():
    def __init__(self, image_size) -> None:
        self.image_size = image_size

    def get_rsna_socre(self):
        pass

    def get_pred_bbox_score(self, coco_result_path, input_size):
        '''
        获取预测结果的边界框和得分
        coco_result_path: coco格式的预测结果存储路径, x_top_left, y_top_left, w, h
        input_size: 预测时的图像大小
        bbox 按照x_top_left, y_top_left, w, h存储
        '''
        with open(coco_result_path) as f:
            self.original_result = json.load(f)
        
        self.pred_bbox = dict()
        self.pred_score = dict()
        for record in self.original_result:
            image_id = record["image_id"]
            category_id = record["category_id"]
            bbox = record["bbox"]
            score = record["score"]
            # 转换为512尺寸下的标注
            if not input_size == self.image_size:
                for i in range(len(bbox)):
                    bbox[i] = bbox[i] * self.image_size  / input_size
            # 将x_top_left, y_top_left转换为 x_center y_center
            if image_id not in self.pred_bbox:
                self.pred_bbox[image_id] = [bbox]
                self.pred_score[image_id] = [score]
            else:
                self.pred_bbox[image_id].append(bbox)
                self.pred_score[image_id].append(score)

        del self.original_result
        return len(self.pred_bbox)

    def get_ground_truth_label(self, catch_path):
        '''
        catch_path: yolov5的标注数据路径，标注格式为x_center y_center w h
        ground_truth_label: 按照x_top_left, y_top_left, w, h存储
        '''
        with open(catch_path, 'rb') as f:
            self.ground_truth_data = pickle.load(f)
        self.ground_truth_label = dict()
        for record in self.ground_truth_data:
            if record[-4:] == '.png':
                image_id = record.split("/")[-1][:-4]
                bbox_array = self.ground_truth_data[record][0]
                image_size = self.ground_truth_data[record][1][0]  # rsna数据集的大小为1024
                bbox_array = bbox_array * image_size
                bbox_array = bbox_array[:, 1:]
                # print(bbox_array)
                bbox_array[:, :2] -= bbox_array[:, 2:] / 2  # 转换center → top_left
                # print(bbox_array)
                # print()
                if not image_size == self.image_size:
                    bbox_array = bbox_array * self.image_size / image_size
                self.ground_truth_label[image_id] = list(bbox_array)
        del self.ground_truth_data
        return len(self.ground_truth_label)

    def get_score(self):
        thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.28, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0]
        all_scores = []
        # check range of thresholds
        for threshold in thresholds:
            threshold_scores = []
            TP = 0
            FP = 0
            TN = 0
            FN = 0
            for image_id in self.ground_truth_label:
                gt_boxes = np.array(self.ground_truth_label[image_id])
                if image_id in self.pred_bbox:
                    pred_boxes = np.array(self.pred_bbox[image_id])
                    pred_scores = np.array(self.pred_score[image_id])
                else:
                    pred_boxes = np.array([])
                    pred_scores = np.array([])

                # if len(pred_scores):
                #     pred_scores[pred_scores < pred_scores[0] * 0.5] = 0.0
                mask = pred_scores * 5 > threshold

                if not len(gt_boxes):  # 标签为空，预测存在任意值则得分为0
                    if np.any(mask):
                        threshold_scores.append(0.0)
                        FP += 1
                    else:
                        TN += 1
                else:
                    if len(pred_scores[mask]) == 0:  # 预测为空，则得分为0
                        FN += 1
                        score = 0.0
                    else:
                        TP += 1
                        score = metric.map_iou(
                            boxes_true=gt_boxes,
                            boxes_pred=pred_boxes[mask],
                            scores=pred_scores[mask],
                        )
                        # print(score)
                    threshold_scores.append(score)

            print("threshold {}, score {}".format(threshold, np.mean(threshold_scores)))
            print(f"TP:{TP} FP:{FP} TN:{TN} FN:{FN}")
            all_scores.append(np.mean(threshold_scores))
        best_score = np.max(all_scores)
        print(f"best_score: {best_score}")
        
        return all_scores

    def visual_result(self, visual_num):
        for image_id in self.pred_bbox:
            pred_bboxs = self.pred_bbox[image_id]
            scores = self.pred_score[image_id]
            gt_bboxs = self.ground_truth_label[image_id]
            # 创建一个宽512高512的黑色画布，RGB(0,0,0)即黑色
            image = np.zeros((512,512,3), np.uint8)
            #写字,字体选择
            font=cv2.FONT_HERSHEY_SCRIPT_COMPLEX
            cv2.putText(image,"pred bbox",(50,50),font,1,(0,255,0),2)
            cv2.putText(image,"GT bbox",(50,100),font,1,(255,0,0),2)
            for box, score in zip(pred_bboxs, scores):
                x1, y1, x2, y2 = round(box[0]), round(box[1]), round(box[0]+box[2]), round(box[1]+box[3])
                # 画矩形，图片对象，左上角坐标，右下角坐标，颜色，宽度
                cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),3)
                # 图片对象，要写的内容，左边距，字的底部到画布上端的距离，字体，大小，颜色，粗细
                cv2.putText(image,f"{score}",(x1,y1),font,1,(0,255,0),2)

            for box in gt_bboxs:
                x1, y1, x2, y2 = round(box[0]), round(box[1]), round(box[0]+box[2]), round(box[1]+box[3])
                # 画矩形，图片对象，左上角坐标，右下角坐标，颜色，宽度
                cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,0),3)
            cv2.imshow(f"{image_id}",image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    yolov5_result_path = r"G:\03_research_file\titan_mirror\kaggle-rsna\yolov5_result\best_predictions.json"
    gt_cache_path = r"G:\03_research_file\titan_mirror\kaggle-rsna\yolov5_result\archive\data.pkl"

    yolov5_test = TestCOCO(image_size=512)
    yolov5_test.get_pred_bbox_score(yolov5_result_path, 1024)
    yolov5_test.get_ground_truth_label(gt_cache_path)
    # yolov5_test.visual_result(100)
    yolov5_test.get_score()

