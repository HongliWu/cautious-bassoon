import pickle as p
import numpy as np
import matplotlib.pyplot as plt
import csv
from src.pytorch_retinanet.anchors import generate_anchors, shift


def score_print(score_path):
    # load score data
    with open(score_path, 'rb') as score_file:
        score_data = p.load(score_file)
        # print score
        # print(score_data)
        tresholds = score_data['tresholds']
        epochs = score_data['epochs']
        scores = score_data['scores']
        print(tresholds)
        print(epochs)
        for epoch in range(len(epochs)):
            for threshold in range(len(tresholds)):
                print(scores[epoch, threshold], end=',')
            print('')


def bbox_x1x2_mask(mask_size, bbox_x1x2):
    mask_row = np.zeros(mask_size, dtype=np.int)
    mask_column = np.zeros(mask_size, dtype=np.int).T
    x1 = int(bbox_x1x2[0] * 2)
    y1 = int(bbox_x1x2[1] * 2)
    x2 = int(bbox_x1x2[2] * 2)
    y2 = int(bbox_x1x2[3] * 2)
    mask_row[y1:y2] = 1
    mask_column[x1:x2] = 1
    mask_column = mask_column.T
    mask = mask_row * mask_column
    return mask


def oof_visualization(oof_path, threshold=0.4, result_dir='./'):
    # load oof data
    with open(oof_path, 'rb') as oof_file:
        oof_data = p.load(oof_file)
    # plot gt boxes on image
    gt_boxes = oof_data['gt_boxes']
    gt_overlay = np.zeros([1024,1024], dtype=np.int)
    for gt_boxes_batch in gt_boxes:
        gt_boxes_image = gt_boxes_batch[0]
        for gt_box_image in gt_boxes_image:
            if gt_box_image[4] == 0.0:
                bbox_x1x2 = gt_box_image[:4]
                gt_box_mask = bbox_x1x2_mask([1024, 1024], bbox_x1x2)
                gt_overlay = gt_overlay + gt_box_mask
    plt.figure()
    plt.imshow(gt_overlay > 1)
    plt.title('gt_overlay_image_area')
    # plt.colorbar()
    # plt.waitforbuttonpress()
    plt.savefig(result_dir + 'gt_overlay_image_area.png')

    boxes = oof_data['boxes']
    scores = oof_data['scores']
    batch_num = len(boxes)
    predict_overlay = np.zeros([1024,1024], dtype=np.int)
    for batch_idx in range(batch_num):
        boxes_image = boxes[batch_idx]
        scores_image = scores[batch_idx]
        box_num = len(scores_image)
        # same as train runner
        if len(scores_image):
            scores_image[scores_image < scores_image[0] * 0.5] = 0.0
        for box_idx in range(box_num):
            if scores_image[box_idx] * 5 > threshold:
                bbox_x1x2 = boxes_image[box_idx]
                predict_box_mask = bbox_x1x2_mask([1024, 1024], bbox_x1x2)
                predict_overlay = predict_overlay + predict_box_mask

    plt.figure()
    plt.imshow(predict_overlay > 1)
    plt.title('predict_overlay_image_area')
    # plt.colorbar()
    # plt.waitforbuttonpress()
    plt.savefig(result_dir + 'predict_overlay_image_area.png')


def oof_predict_offset(oof_path, threshold=0.4, result_dir='./', ratio=[0.5, 1, 2]):
    # load oof data
    with open(oof_path, 'rb') as oof_file:
        oof_data = p.load(oof_file)

    gt_boxes = oof_data['gt_boxes']
    gt_boxes_x = list()
    gt_boxes_y = list()
    gt_boxes_w = list()
    gt_boxes_h = list()
    for gt_boxes_batch in gt_boxes:
        gt_boxes_image = gt_boxes_batch[0]
        for box in gt_boxes_image:
            if box[4] == 0:
                gt_boxes_x.append([box[0]])
                gt_boxes_y.append([box[1]])
                gt_boxes_w.append([box[2] - box[0]])
                gt_boxes_h.append([box[3] - box[1]])

    boxes = oof_data['boxes']
    scores = oof_data['scores']
    predict_boxes_x = list()
    predict_boxes_y = list()
    predict_boxes_w = list()
    predict_boxes_h = list()
    predict_score = list()
    for predict_boxes_image, predict_score_image in zip(boxes, scores):
        if len(predict_score_image):
            predict_score_image[predict_score_image < predict_score_image[0] * 0.5] = 0.0
        for box_index in range(len(predict_score_image)):
            if predict_score_image[box_index] * 5 > threshold:
                box = predict_boxes_image[box_index]
                predict_boxes_x.append([box[0]])
                predict_boxes_y.append([box[1]])
                predict_boxes_w.append([box[2] - box[0]])
                predict_boxes_h.append([box[3] - box[1]])
                predict_score.append(predict_score_image[box_index])

    plt.figure(figsize=(12,12))
    plt.scatter(gt_boxes_x, gt_boxes_y)
    plt.scatter(predict_boxes_x, predict_boxes_y)
    plt.xlim(0, 512)
    plt.ylim(512, 0)
    plt.title('left up point scatter of gt_bbox and predict of oof 010')
    plt.xlabel('position x of bbox')
    plt.ylabel('position y of bbox')
    plt.legend(['gt_boxes', 'predict_boxes'])
    # plt.waitforbuttonpress()
    plt.savefig(result_dir + 'position_scatter.png')
    plt.figure(figsize=(12,12))
    plt.plot([0, 1000], [0, 1000 * ratio[0]])
    plt.plot([0, 1000], [0, 1000 * ratio[1]])
    plt.plot([0, 1000], [0, 1000 * ratio[2]])
    plt.scatter(gt_boxes_w, gt_boxes_h)
    plt.scatter(predict_boxes_w, predict_boxes_h)

    # plot anchor point
    pyramid_levels = [2, 3, 4, 5, 6, 7]
    # strides = [2 ** x for x in pyramid_levels] * 2
    sizes = [2 ** (x + 2) for x in pyramid_levels]
    ratios = np.array(ratio)
    scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
    all_anchors = np.zeros((0, 4)).astype(np.float32)
    for idx, _ in enumerate(pyramid_levels):
        anchors = generate_anchors(base_size=sizes[idx], ratios=ratios, scales=scales)
        all_anchors = np.append(all_anchors, anchors, axis=0)
    anchors_w = all_anchors[:, 2] * 2
    anchors_h = all_anchors[:, 3] * 2
    plt.scatter(anchors_w, anchors_h)

    plt.xlim(0, 450)
    plt.ylim(0, 450)
    plt.title('size scatter of gt_bbox and predict of oof 010')
    plt.xlabel('width of bbox')
    plt.ylabel('height of bbox')
    plt.legend(['y=1/2 x', 'y=x', 'y=2x', 'gt_boxes', 'predict_boxes', 'anchors'])
    # plt.waitforbuttonpress()
    plt.savefig(result_dir + 'size_scatter.png')


def csv_label_statistics(csv_label_path, result_dir='./'):
    with open(csv_label_path, 'r') as csv_label_file:
        label_reader = csv.reader(csv_label_file)

        gt_boxes_x = list()
        gt_boxes_y = list()
        gt_boxes_w = list()
        gt_boxes_h = list()

        for row in label_reader:
            if row[5] == '1':
                gt_boxes_x.append([eval(row[1])])
                gt_boxes_y.append([eval(row[2])])
                gt_boxes_w.append([eval(row[3])])
                gt_boxes_h.append([eval(row[4])])

        plt.figure(figsize=(12, 12))
        plt.scatter(gt_boxes_x, gt_boxes_y)
        plt.xlim(0, 1024)
        plt.ylim(1024, 0)
        plt.title('left up point scatter of gt_bbox of stage_2 train label')
        plt.xlabel('position x of bbox')
        plt.ylabel('position y of bbox')
        plt.legend(['gt_boxes'])
        # plt.waitforbuttonpress()
        plt.savefig(result_dir + 'stage2_train_label_position_scatter.png')
        plt.figure(figsize=(12, 12))
        plt.plot([0, 1000], [0, 500])
        plt.plot([0, 1000], [0, 1000])
        plt.plot([0, 1000], [0, 2000])
        plt.scatter(gt_boxes_w, gt_boxes_h)
        plt.xlim(0, 1000)
        plt.ylim(0, 1000)
        plt.title('size scatter of gt_bbox of stage_2 train label')
        plt.xlabel('width of bbox')
        plt.ylabel('height of bbox')
        # plot anchor point
        pyramid_levels = [2, 3, 4, 5, 6, 7]
        # strides = [2 ** x for x in pyramid_levels] * 2
        sizes = [2 * 2 ** (x + 2) for x in pyramid_levels]
        ratios = np.array([0.5, 1, 2])
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
        all_anchors = np.zeros((0, 4)).astype(np.float32)
        for idx, _ in enumerate(pyramid_levels):
            anchors = generate_anchors(base_size=sizes[idx], ratios=ratios, scales=scales)
            all_anchors = np.append(all_anchors, anchors, axis=0)
        anchors_w = all_anchors[:, 2] * 2
        anchors_h = all_anchors[:, 3] * 2
        plt.scatter(anchors_w, anchors_h)
        plt.legend(['y=1/2 x', 'y=x', 'y=2x', 'gt_boxes', 'anchors'])
        plt.savefig(result_dir + 'stage2_train_label_size_scatter.png')


def ratio_w_h_statistics(csv_label_path, result_dir='./'):
    with open(csv_label_path, 'r') as csv_label_file:
        label_reader = csv.reader(csv_label_file)

        gt_boxes_x = list()
        gt_boxes_y = list()
        gt_boxes_w = list()
        gt_boxes_h = list()
        gt_boxes_ratio = list()
        gt_boxes_area = list()

        for row in label_reader:
            if row[5] == '1':
                gt_boxes_x.append([eval(row[1])])
                gt_boxes_y.append([eval(row[2])])
                gt_boxes_w.append([eval(row[3])])
                gt_boxes_h.append([eval(row[4])])
                gt_boxes_ratio.append([eval(row[4]) / eval(row[3])])
                gt_boxes_area.append([eval(row[4]) * eval(row[3])])

        gt_boxes_w = np.array(gt_boxes_w).squeeze(1)
        gt_boxes_h = np.array(gt_boxes_h).squeeze(1)
        gt_boxes_ratio = np.array(gt_boxes_ratio).squeeze(1)
        gt_boxes_area = np.array(gt_boxes_area).squeeze(1)

        w_max = gt_boxes_w.max()
        w_min = gt_boxes_w.min()
        h_max = gt_boxes_h.max()
        h_min = gt_boxes_h.min()
        ratio_max = gt_boxes_ratio.max()
        ratio_min = gt_boxes_ratio.min()
        area_max = gt_boxes_area.max()
        area_min = gt_boxes_area.min()

        plt.figure(figsize=(16, 16))
        plt.subplot(4, 1, 1)
        n, bins, patches = plt.hist(gt_boxes_w, bins=20, facecolor="blue", edgecolor="black", alpha=0.7)
        for x, y in zip(bins, n):
            plt.text(x, y+10, str(int(y)))
        plt.xticks(bins)
        plt.title('histogram of width' + '   max is:{:.2f} min is:{:.2f}'.format(w_max, w_min))
        # plt.waitforbuttonpress()
        # plt.savefig(result_dir + 'histogram_of_width_of_stage2_train_label.png')

        plt.subplot(4, 1, 2)
        # plt.figure(figsize=(6, 8))
        n, bins, patches = plt.hist(gt_boxes_h, bins=20, facecolor="blue", edgecolor="black", alpha=0.7)
        for x, y in zip(bins, n):
            plt.text(x, y + 10, str(int(y)))
        plt.xticks(bins)
        plt.title('histogram of height' + '   max is:{:.2f} min is:{:.2f}'.format(h_max, h_min))
        # plt.waitforbuttonpress()
        # plt.savefig(result_dir + 'histogram_of_height_of_stage2_train_label.png')

        # plt.figure(figsize=(6, 8))
        plt.subplot(4, 1, 3)
        n, bins, patches = plt.hist(gt_boxes_ratio, bins=20, facecolor="blue", edgecolor="black", alpha=0.7)
        for x, y in zip(bins, n):
            plt.text(x, y + 10, str(int(y)))
        plt.xticks(bins)
        plt.title('histogram of ratio' + '   max is:{:.2f} min is:{:.2f}'.format(ratio_max, ratio_min))
        # plt.waitforbuttonpress()
        # plt.savefig(result_dir + 'histogram_of_ratio_of_stage2_train_label.png')
        # plt.figure(figsize=(6, 8))

        plt.subplot(4, 1, 4)
        n, bins, patches = plt.hist(gt_boxes_area, bins=20, facecolor="blue", edgecolor="black", alpha=0.7)
        for x, y in zip(bins, n):
            plt.text(x, y + 10, str(int(y)))
        plt.xticks(bins)
        plt.title('histogram of area' + '   max is:{:.2f} min is:{:.2f}'.format(area_max, area_min))
        # plt.waitforbuttonpress()
        # plt.savefig(result_dir + 'histogram_of_ratio_of_stage2_train_label.png')
        plt.savefig(result_dir + 'histogram_of_stage2_train_label.png')


def split_ratio(csv_label_path, result_dir='./'):
    with open(csv_label_path, 'r') as csv_label_file:
        label_reader = csv.reader(csv_label_file)

        gt_boxes_x = list()
        gt_boxes_y = list()
        gt_boxes_w = list()
        gt_boxes_h = list()
        gt_boxes_ratio = list()
        gt_boxes_area = list()

        for row in label_reader:
            if row[5] == '1':
                gt_boxes_x.append([eval(row[1])])
                gt_boxes_y.append([eval(row[2])])
                gt_boxes_w.append([eval(row[3])])
                gt_boxes_h.append([eval(row[4])])
                gt_boxes_ratio.append([eval(row[4]) / eval(row[3])])
                gt_boxes_area.append([eval(row[4]) * eval(row[3])])

        gt_boxes_w = np.array(gt_boxes_w).squeeze(1)
        gt_boxes_h = np.array(gt_boxes_h).squeeze(1)
        gt_boxes_ratio = np.array(gt_boxes_ratio).squeeze(1)
        gt_boxes_area = np.array(gt_boxes_area).squeeze(1)

        box_num = gt_boxes_ratio.shape[0]
        gt_boxes_ratio.sort()
        ratio_split_1 = gt_boxes_ratio[0:int(box_num / 3)]
        ratio_split_2 = gt_boxes_ratio[int(box_num / 3):int(2 * box_num / 3)]
        ratio_split_3 = gt_boxes_ratio[int(2 * box_num / 3):]

        ratio_1 = ratio_split_1.mean()
        ratio_2 = ratio_split_2.mean()
        ratio_3 = ratio_split_3.mean()

        plt.figure(figsize=(6, 6))
        plt.plot([0, 1000], [0, 1000 * ratio_1], 'r')
        plt.plot([0, 1000], [0, 1000 * ratio_2], 'tomato')
        plt.plot([0, 1000], [0, 1000 * ratio_3], 'coral')
        plt.plot([0, 1000], [0, 1000 * 0.5], 'g')
        plt.plot([0, 1000], [0, 1000 * 1], 'lime')
        plt.plot([0, 1000], [0, 1000 * 2], 'lightgreen')
        plt.scatter(gt_boxes_w, gt_boxes_h, s=1)
        plt.xlim(0, 1000)
        plt.ylim(0, 1000)
        plt.title('new ratio from three parts mean')
        plt.xlabel('width of bbox')
        plt.ylabel('height of bbox')
        plt.legend([f'h={ratio_1:.5f}*w', f'h={ratio_2:.5f}*w', f'h={ratio_3:.5f}*w',
                    'h=0.5*w', 'h=w', 'h=2*w', 'gt_boxes'])
        plt.savefig(result_dir + 'new_ratio_from_three_parts_mean.png')


if __name__ == "__main__":
    oof_path = '../src/output/changed_ratio/oof/se_resnext101_dr0.75_512_fold_0/010.pkl'
    save_path = './changed_ratio/'
    stage_2_train_path = '../annotations/stage_2_train_labels.csv'
    score_pkl_path = '../src/output/changed_ratio/scores/se_resnext101_dr0.75_512_fold_0/scores.pkl'
    ratio = [0.89, 1.45, 2.16]
    threshold = 0.4
    # csv_label_statistics(stage_2_train_path, save_path)
    oof_visualization(oof_path, threshold, save_path)
    oof_predict_offset(oof_path, threshold, save_path, ratio=ratio)
    # ratio_w_h_statistics(stage_2_train_path, save_path)
    # split_ratio(stage_2_train_path, save_path)
    # score_print(score_pkl_path)
