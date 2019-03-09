import darknet as dn
import ctypes
import numpy as np
import cv2

class YOLO:
    '''
    YOLO Model Class
    '''
    def __init__(self):
        self.__data = None
        self.__cfg = None
        self.__weights = None
        self.__net = None
        self.__meta = None
        self.__net_size = None

    def free_network(self):
        '''
        ネットワークを開放する
        '''
        if self.__net is not None:
            dn.free_network(self.__net)
            self.__net = None
        else:
            raise Exception("Free network exception: network is None")
        
    def load_network(self, data, cfg, weights):
        '''
        YOLOの関連ファイルパスを受け取って、ネットワークをロードする。

        Parameters
        ----------
        data : string
            dataファイルパス
        cfg : string
            cfgファイルパス
        weights : string
            weightsファイルパス
        '''
        if self.__net is None:
            self.__data = data.encode('utf-8')
            self.__cfg = cfg.encode('utf-8')
            self.__weights = weights.encode('utf-8')

            self.__net = dn.load_net(self.__cfg, self.__weights, 0)
            self.__meta = dn.load_meta(self.__data)
            self.__net_size = (dn.lib.network_width(self.__net), dn.lib.network_height(self.__net))
        else:
            raise Exception("Load network exception: network is not None. Please execute free_network()")

    def detect(self, img, threshold=0.24):
        if self.__net is not None:
            net_w = self.__net_size[0]
            net_h = self.__net_size[1]

            # レターボックス形式に変換する
            letter_box, st_x, st_y = YOLO.get_letter_box(img, net_w, net_h)

            # ctypesで受け渡せる形式に変換する
            darknet_img_data = YOLO.get_darknet_img_data(letter_box)
            args_data = darknet_img_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            img_dn_lb = dn.IMAGE(w = net_w, h = net_w, c = 3, data = args_data)

            # 物体検出を実行する
            result = dn.detect(self.__net, self.__meta, img_dn_lb, threshold)

            org_size = (img.shape[1], img.shape[0])
            lb_size = (letter_box.shape[1], letter_box.shape[0])
            # レターボックスの座標系からオリジナルの座標系に変換する
            result = YOLO.get_original_coords_result(result, org_size, lb_size, st_x, st_y)

            return result
        else:
            raise Exception("execute detection exception: network is None")

    def get_data(self):
        return self.__data
    def get_cfg(self):
        return self.__cfg
    def get_meta(self):
        return self.__meta
    def get_net_size(self):
        return self.__net_size

    @staticmethod
    def get_letter_box(img, lw, lh):
        '''
        画像をレターボックス形式に変換する

        Parameters
        ----------
        img : ndarray
            BGR画像

        lw : int
            レターボックス変換後の幅

        lh : int
            レターボックス変換後の高さ

        Returns
        -------
        letter_box : ndarray
            レターボックス画像
        w_ratio : float
            元画像とレターボックス画像の幅の比率（レターボックス÷元画像）
        h_ratio : float
            元画像とレターボックス画像の高さの比率（レターボックス÷元画像）
        st_x : int
            元画像の開始x座標
        st_y : int
            元画像の開始y座標

        '''

        imh, imw, c = img.shape
        w_ratio = lw/float(imw)
        h_ratio = lh/float(imh)

        # 縦横どちらに合わせるか決定する
        if w_ratio < h_ratio:
            new_w = lw
            new_h = int((imh * lw)/imw)
        else:
            new_h = lh
            new_w = int((imw * lh)/imh)

        # 画像をレターボックスにはまるようにリサイズする
        resized_img = cv2.resize(img, (new_w, new_h))

        # レターボックスのキャンバスを灰色埋めで作成する
        letter_box = np.full([lh, lw, c], 127, dtype=np.uint8)

        # 画像の開始位置を決定する
        st_x = int((lw - new_w)/2)
        st_y = int((lh - new_h)/2)

        letter_box[st_y:(st_y+new_h), st_x:(st_x+new_w), :] = resized_img
        
        return letter_box, st_x, st_y

    @staticmethod
    def get_darknet_img_data(img):
        '''
        darknet形式の一次元画像データを取得する

        Parameters
        ----------
        img : ndarray
            BGR画像

        Returns
        -------
        darknet_img_data : ndarray
            darknet形式の一次元画像データ

        '''

        h, w, c = img.shape
        blues = img[:,:, 0].reshape(w*h)
        greens = img[:,:, 1].reshape(w*h)
        reds = img[:,:, 2].reshape(w*h)
        darknet_img_data = np.concatenate((reds, greens, blues)).astype(np.float32)/255.0

        return darknet_img_data
    
    @staticmethod
    def get_original_coords_result(letter_box_coords_result, org_size, lb_size, st_x, st_y):
        '''
        レターボックス座標系からオリジナルの座標系の結果を取得する
        '''

        original_coords_result = []

        for lrb in letter_box_coords_result:
            # 座標を取得する
            min_x = lrb[2]
            min_y = lrb[3]
            max_x = lrb[4]
            max_y = lrb[5]

            original_coords_result.append([lrb[0], lrb[1], YOLO.get_original_coords_rect((min_x, min_y, max_x, max_y), org_size, lb_size, st_x, st_y)])

        return original_coords_result

    @staticmethod
    def get_original_coords_rect(letter_box_coords_rect, org_size, lb_size, st_x, st_y):
        # レターボックス内の画像とオリジナル画像の比率を取得
        org_size_in_letter_box = (lb_size[0] - st_x*2, lb_size[1] - st_y*2)

        w_ratio = org_size[0] / org_size_in_letter_box[0]
        h_ratio = org_size[1] / org_size_in_letter_box[1]

        # 座標を取得する
        min_x = (letter_box_coords_rect[0] - st_x) * w_ratio
        min_y = (letter_box_coords_rect[1] - st_y) * h_ratio
        max_x = (letter_box_coords_rect[2] - st_x) * w_ratio
        max_y = (letter_box_coords_rect[3] - st_y) * h_ratio


        min_x = min_x if min_x > 0 else 0
        min_y = min_y if min_y > 0 else 0
        max_x = max_x if max_x < org_size[0] else org_size[0] - 1
        max_y = max_y if max_y < org_size[1] else org_size[1] - 1

        return (int(min_x), int(min_y), int(max_x), int(max_y))

if __name__ == "__main__":
    model = YOLO()

    data_path = "/home/abe/dlb2018-team14/darknet/cfg/yolov3.data"
    cfg_path = "/home/abe/dlb2018-team14/darknet/cfg/yolov3.cfg"
    weight_path = "/home/abe/dlb2018-team14/darknet/yolov3.weights"

    model.load_network(data_path, cfg_path, weight_path)

    img = cv2.imread("/home/abe/dlb2018-team14/darknet/data/dog.jpg")

    ret = model.detect(img)

    for r in ret:
        cv2.rectangle(img, (r[2][0], r[2][1]), (r[2][2], r[2][3]), (0,0,255))

    #cv2.imwrite("./test.jpg", img)

    h, w, _ = img.shape
    f = open("./test.txt", "w")
    for r in ret:
        class_id = r[0]
        prob = r[1]        
        rect_width = (r[2][2] - r[2][0])
        rect_height = (r[2][3] - r[2][1])
        center_x = (r[2][0] + int(rect_width/2))
        center_y = (r[2][1] + int(rect_height/2))

        rect_width = rect_width/float(w)
        rect_height = rect_height/float(h)
        center_x = center_x/float(w)
        center_y = center_y/float(h)

        f.write("{:d},{:f},{:f},{:f},{:f},{:f}\n".format(class_id, prob, center_x, center_y, rect_width, rect_height))
    f.close()