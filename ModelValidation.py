# original
from YOLO import YOLO
import Constant as const

# builtin
import os
import re
import sys

# requrement
import cv2
import numpy as np
import matplotlib.pyplot as plt

class ModelValidation:
    def __init__(self, model=None, test_data_dir=None):
        self.__model = model
        self.__test_data_dir = test_data_dir

    def get_validation_result_with_step(self, threshold_step=0.1, start_threshold=0.1, end_threshold=0.9):
        thresholds = [start_threshold]

        add_threshold = start_threshold
        while True:
            add_threshold += threshold_step

            if add_threshold > end_threshold:
                thresholds.append(end_threshold)
                break
            else:
                thresholds.append(add_threshold)

        macro_result, micro_result = self.get_validation_result(thresholds[0])
        macro_result = np.array([macro_result])
        micro_result = np.array([micro_result])

        for i in range(1, len(thresholds)):
            macro_tmp_result, micro_tmp_result = self.get_validation_result(thresholds[i])
            macro_result = np.concatenate((macro_result, np.array([macro_tmp_result])))
            micro_result = np.concatenate((micro_result, np.array([micro_tmp_result])))

        return macro_result, micro_result

    def get_validation_result(self, threshold=0.24):
        fns = os.listdir(self.__test_data_dir)
        fns.sort()

        macro_precision = 0
        macro_recall = 0
        macro_f_measure = 0

        precision_count = 0
        recall_count = 0
        predict_label_count = 0
        gt_label_count = 0
        
        test_data_num = 0
        for fn in fns:
            tmp = fn.split(".")
            basename = tmp[0]
            ext = tmp[1]

            if re.match(r'jpg|JPG|jpeg|JPEG|png|PNG', ext):
                img = cv2.imread(os.path.join(self.__test_data_dir, fn))
                gt_label = ModelValidation.read_yolo_label(os.path.join(self.__test_data_dir, basename+".txt"))
            else:
                continue

            predict_label = self.__model.detect(img, threshold)
            h, w, _ = img.shape
            predict_label = ModelValidation.convert_result_to_float(predict_label, (w, h))

            pre_rec_f, pre_rec_count = ModelValidation.get_pre_rec_f(gt_label, predict_label)

            macro_precision += pre_rec_f[0]
            macro_recall += pre_rec_f[1]
            macro_f_measure += pre_rec_f[2]

            precision_count += pre_rec_count[0]
            recall_count += pre_rec_count[1]
            predict_label_count += len(predict_label)
            gt_label_count += len(gt_label)

            test_data_num += 1

        macro_precision = macro_precision / test_data_num
        macro_recall = macro_recall / test_data_num
        macro_f_measure = macro_f_measure / test_data_num
        micro_precision = precision_count / float(predict_label_count) if predict_label_count!=0 else 0
        micro_recall = recall_count / float(gt_label_count) if gt_label_count!=0 else 0

        if micro_precision + micro_recall == 0:
            micro_f_measure = 0
        else:  
            micro_f_measure = (2 * micro_precision * micro_recall) / float(micro_precision + micro_recall)

        return (macro_precision, macro_recall, macro_f_measure), (micro_precision, micro_recall, micro_f_measure)

    # getter and setter
    def set_model(self, model):
        self.__model = model
    def set_test_data_dir(self, test_data_dir):
        self.__test_data_dir = test_data_dir
    def get_model(self):
        return self.__model
    def get_test_data_dir(self):
        return self.__test_data_dir

    @staticmethod
    def read_yolo_label(file_path):
        labels = []
        with open(file_path) as f:
            for line in f:
                line = line.strip()

                tmp = line.split(",")

                class_id = int(tmp[0])
                prob = float(tmp[1])
                center_x = float(tmp[2])
                center_y = float(tmp[3])
                width = float(tmp[4])
                height = float(tmp[5])

                min_x = center_x - width/2.0
                min_y = center_y - height/2.0
                max_x = center_x + width/2.0
                max_y = center_y + height/2.0

                labels.append([class_id, prob, min_x, min_y, max_x, max_y])

        return labels

    @staticmethod
    def convert_result_to_float(result, img_size):
        yolo_style_label = []
        w, h = img_size

        for r in result:
            class_id = r[0]
            prob = r[1]
            min_x = r[2][0]/float(w)
            min_y = r[2][1]/float(h)
            max_x = r[2][2]/float(w)
            max_y = r[2][3]/float(h)

            yolo_style_label.append([class_id, prob, min_x, min_y, max_x, max_y])

        return yolo_style_label

    @staticmethod
    def get_pre_rec_f(gt_label, predict_label, iou_threshold=0.6):
        if len(predict_label) == 0:
            precision = 0
            recall = 0
            f_measure = 0
            precision_count = 0
            recall_count = 0

            return (precision, recall, f_measure), (precision_count, recall_count)

        # 適合率の計算
        # 検出した物体が、一定値以上のIOUで正解の物体と重なっているか
        precision_count = 0
        for i, pl in enumerate(predict_label):
            for j, gtl in enumerate(gt_label):
                if gtl[0] == pl[0]:
                    iou = ModelValidation.calc_iou(gtl[2:6], pl[2:6])
                    if iou > iou_threshold:
                        precision_count += 1
                        break
        precision = precision_count / float(len(predict_label))

        # 再現率の計算
        # 正解の物体が、一定値以上のIOUで１つ以上検出されているかどうか
        recall_count = 0
        for i, gtl in enumerate(gt_label):
            for j, pl in enumerate(predict_label):
                if gtl[0] == pl[0]:
                    iou = ModelValidation.calc_iou(gtl[2:6], pl[2:6])
                    if iou > iou_threshold:
                        recall_count += 1
                        break
        recall = recall_count / float(len(gt_label))
        
        # F値の計算
        if precision + recall == 0:
            f_measure = 0
        else:  
            f_measure = (2 * precision * recall) / float(precision + recall)

        return (precision, recall, f_measure), (precision_count, recall_count)

    @staticmethod
    def calc_iou(a, b):
        min_x = max([a[0], b[0]])
        max_x = min([a[2], b[2]])
        
        min_y = max([a[1], b[1]])
        max_y = min([b[3], b[3]])
        
        share_box_width = max_x - min_x
        share_box_height = max_y - min_y
        
        if share_box_width > 0 and share_box_height > 0:
            area = share_box_width * share_box_height
        else:
            area = 0
            
        a_area = (a[2] - a[0]) * (a[3] - a[1])
        b_area = (b[2] - b[0]) * (b[3] - b[1])
        
        iou = area / float(a_area + b_area - area)
        
        return iou

    @staticmethod
    def output_plot(output_path, data):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        # プロット
        ax.plot(data[:, 1], data[:, 0])

        # 軸ラベル設定
        ax.set_xlabel('recall')
        ax.set_ylabel('precision')
        ax.set_title('Precision Recall Curve')

        # 範囲を指定
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])

        # 保存
        plt.savefig(output_path)


if __name__=="__main__":
    data_path = const.DATA_FILE_PATH
    cfg_path = const.CFG_FILE_PATH
    weight_path = const.WEIGHT_FILE_PATH

    test_data_dir = const.TEST_DATA_DIR_PATH

    # 検証したいモデルを用意
    model = YOLO()
    model.load_network(data_path, cfg_path, weight_path)

    # 検証クラスを準備
    validater = ModelValidation()
    
    validater.set_model(model)
    validater.set_test_data_dir(test_data_dir)

    start_threshold = 0.01
    threshold_step = 0.05
    end_threshold = 0.99
    macro_result, micro_result = validater.get_validation_result_with_step(threshold_step, start_threshold, end_threshold)

    validater.output_plot("./macro.jpg", macro_result[:, 0:2])
    validater.output_plot("./micro.jpg", micro_result[:, 0:2])

    print("Max macro F-measure: {:.4f}".format(macro_result[:, 0:2].max()))
    print("Max micro F-measure: {:.4f}".format(micro_result[:, 0:2].max()))
    