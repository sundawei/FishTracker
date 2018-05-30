import os
import cv2
import math
import numpy as np
from time import time
from image_preprocess import *
from ransac import multi_seq_ransac

np.random.seed(1234)
# parameters setting for fish ransac
min_fish_length = 40   # min body length of a fish
max_fish_length = 150  # max body length of a fish
min_score = min_fish_length * 2 / 3  # min consensus set size
extention_percent = 0.25  # the extention percentage
fish_width = 40  # min([min_fish_length - 2, 40])  # the fish body width
easy_pattern_num = min_fish_length * 5  # the max number of points in easy mode


class RotatedRect(object):
    def __init__(self):
        self.center = np.array([0, 0])  # the center of rectangle
        self.width = 0  # short axis, p0-p3
        self.length = 0  # long axis, p0-p1
        self.angle = 0  # the angle between middle axis of rectangle and positive direction of x-axis, ∈ [-90,90)
        self.curvature = 0  # the curved angle of fish body computed from three anchor points
        self.corners = np.zeros((4, 2))  # p0 is the lowest point, p0-p1-p2-p3,counter/clockwise
        self.anchors = np.zeros((3, 2))

    @staticmethod
    def extend_lines(models, ext_percent):
        """ extend fitted lines along fishhead and fishtail directions respectively
        :param models: models
        :param ext_percent: the percentage of segment length to be extended
        :return: extended lines
        """
        for i in range(len(models)):
            for j in range(len(models[i])):
                anchors = models[i][j]['anchors']
                anchors[[0, 2]] += (anchors[[0, 2]] - anchors[1]) * ext_percent
                endpts = models[i][j]['endpts']
                endpts += (endpts - anchors[1]) * ext_percent
                models[i][j]['anchors'] = anchors
                models[i][j]['endpts'] = endpts
        return models

    @staticmethod
    def check_border(rect, img_shape):
        xmin, xmax = np.sort(rect.corners[:, 0])[[0, -1]]
        ymin, ymax = np.sort(rect.corners[:, 1])[[0, -1]]
        if (xmin >= 0) & (xmax < img_shape[1]) & (ymin >= 0) & (ymax < img_shape[0]):
            return True
        else:
            return False

    @staticmethod
    def compute_rects(models, img_shape):
        """ taking two endpts of a fitted line as the middel axis line of a rotated rectangle
            compute four vertices, center, and rotation angle of the rotated rectangle.
            Besides, since cropping and padding image changes the origin of original image,
            for keeping coordinates consistent, we need to shift them back by adding offset.
        :param models: a model including 'endpts' and 'anchors' item
        :param img_shape: used for detecting whether a rect is beyond the border of the image from which
                          a corresponding patch will be cropped. The imge_shape is the shape of the image.
        :return: rotated rects
        """
        rects = []
        for i in range(len(models)):
            group_rects = []
            # find 4 corners which form a rectangle with the segment as its middle axis
            for j in range(len(models[i])):
                # the angle between the fitted line and the x-axis positive direction  ∈[-90.90)
                endpts = models[i][j]['endpts']
                # delta_x, delta_y = endpts[1, 0] - endpts[0, 0], endpts[1, 1] - endpts[0, 1]
                # radian = math.atan(delta_y / delta_x)
                radian = round(math.atan2(endpts[1, 1] - endpts[0, 1], endpts[1, 0] - endpts[0, 0]), 3)
                if radian >= round(math.pi / 2, 3):
                    radian = radian - math.pi
                delta = np.array([-math.sin(radian), math.cos(radian)]) * fish_width / 2
                front_side = endpts - delta  # the side face to the x-axis
                back_side = endpts + delta  # the side back to the x-axis
                corners = np.zeros((4, 2))
                corners[:] = front_side[0], front_side[1], back_side[1], back_side[0]
                va = models[i][j]['anchors'][0] - models[i][j]['anchors'][1]
                vb = models[i][j]['anchors'][2] - models[i][j]['anchors'][1]
                curved = np.clip(np.dot(va, vb) / np.linalg.norm(va) / np.linalg.norm(vb), -1, 1)

                rect = RotatedRect()
                rect.center = np.mean(corners, axis=0)  # the centroid of fish body
                rect.width = np.sqrt(np.sum((corners[0] - corners[3]) ** 2))  # the width of fish body
                rect.length = np.sqrt(np.sum((corners[0] - corners[1]) ** 2))  # the length of fish body
                rect.angle = radian * 180 / math.pi  # the orientation of fish body
                rect.corners = corners  # the scope of the entire fish body
                rect.anchors = models[i][j]['anchors']
                rect.curvature = 180 - math.acos(curved) * 180 / math.pi

                if RotatedRect.check_border(rect, img_shape) is False:
                    continue

                rect_list = rect.split_rect(min_fish_length * 2)
                group_rects.extend(rect_list)
            # group_rects = [rect for rect in group_rects if rect != []]
            if len(group_rects) > 0:
                rects.append(group_rects)
        return rects

    @staticmethod
    def crop_rects(image, rects):
        # compute an enclosing rect for each rotatedrect
        patches = []
        for i in range(len(rects)):
            group_patches = []
            for j in range(len(rects[i])):
                xmin, xmax = np.sort(rects[i][j].corners[:, 0])[[0, -1]].astype(np.int)
                ymin, ymax = np.sort(rects[i][j].corners[:, 1])[[0, -1]].astype(np.int)
                img_rect = image[ymin:ymax + 1, xmin:xmax + 1]
                # rotation center is local img_rect center
                h, w = img_rect.shape[0:2]
                angle = rects[i][j].angle  # range between [-90, 90)
                rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
                # compute the image translation
                sin, cos = math.sin(math.radians(angle)), math.cos(math.radians(angle))
                nh, nw = np.abs(np.array([[cos, sin], [sin, cos]])).dot(np.array([h, w])).astype(np.int)
                rotation_matrix[:, 2] += np.array([(nw - w) / 2, (nh - h) / 2])
                # trasnform image
                trans_img = cv2.warpAffine(img_rect, rotation_matrix, (nw, nh), borderValue=(255, 255, 255))
                # transform rotatedrect corners
                trans_corners = rects[i][j].corners - np.array([xmin, ymin])
                vertices = np.hstack((trans_corners, np.ones((4, 1))))
                rotated_corners = np.transpose(np.dot(rotation_matrix, vertices.T))
                # crop image by extracting up-right rectangular region defined by rotated_corners
                xmin, xmax = np.sort(rotated_corners[:, 0])[[0, -1]].astype(np.int)
                ymin, ymax = np.sort(rotated_corners[:, 1])[[0, -1]].astype(np.int)
                patch = np.copy(trans_img[ymin:ymax, xmin:xmax])
                group_patches.append(patch)
            patches.append(group_patches)
        return patches

    def split_rect(self, normal_length):
        """ if rect length is overly long, then split the rect into two,
            and the length of the split rect is obtained by truncating a
            certain ratio of the original one from two ends respectively.

            ratio = -0.8 * (actual_length/normal_length) + 2.2, such that
            when actual/normal = 1.5, split_ratio = 1
                 actual/normal = 1.75, split_ratio = 0.8
                 actual/normal = 2, split_ratio = 0.6
        :param normal_length: the normal length of a rect(a fish) which is regraded
                              as being 2 times of the min rect length allowed
        :return: a list of rect
        """
        def ratio(fish_length):
            return -0.8 * (fish_length / normal_length) + 2.2

        rect_list = []
        # ratio = -0.8 * (self.length / normal_length) + 2.2
        if ratio(self.length) > ratio(min_fish_length):  # rect.length < min_fish_length
            rect_list = []
        elif ratio(self.length) >= ratio(max_fish_length):  # rect.length not long enough to split into two
            rect_list.append(self)
        else:  # length > 1.5 * normal_length
            r = max([0.6, ratio(self.length)])
            split_ratio = np.array([1 - r, r])
            middle1 = split_ratio.dot(self.corners[[0, 1]])
            middle2 = split_ratio.dot(self.corners[[3, 2]])
            middle3 = split_ratio[::-1].dot(self.corners[[0, 1]])
            middle4 = split_ratio[::-1].dot(self.corners[[3, 2]])
            corners1 = np.array([self.corners[0], middle1, middle2, self.corners[3]])
            corners2 = np.array([middle3, self.corners[1], self.corners[2], middle4])

            rect1, rect2 = RotatedRect(), RotatedRect()
            rect1.center, rect1.corners = np.mean(corners1, axis=0), corners1
            rect2.center, rect2.corners = np.mean(corners2, axis=0), corners2
            rect1.width, rect1.length, rect1.angle = self.width, self.length * r, self.angle
            rect2.width, rect2.length, rect2.angle = self.width, self.length * r, self.angle
            rect_list.extend([rect1, rect2])
            # rect_list.extend(rect1.split_rect(normal_length))
            # rect_list.extend(rect2.split_rect(normal_length))
        return rect_list

    @staticmethod
    def shift_rects(rects, offset):
        """the origin of rects are offset when detecting thin_binary image
           now shift them back by adding the offset
           offset = crop_size - pad_size
           rect = rect + offset
        :param rects: detected rects
        :param offset: offset coordinates
        :return: shifted rects
        """
        for grp_rect in rects:
            for rect in grp_rect:
                rect.center += offset
                rect.corners += offset
                rect.anchors += offset
        return rects

    @staticmethod
    def draw_rects(image, rects, offset, txt=''):
        # draw rectangles in the image
        for i in range(len(rects)):
            for j in range(len(rects[i])):
                corners = (rects[i][j].corners + offset).astype(int)
                corners = [tuple(x) for x in corners.tolist()]
                color = tuple(np.random.randint(0, 250, size=(3,)).tolist())
                for m in range(len(corners)):
                    cv2.line(image, corners[m - 1], corners[m], color, 1)
        # put description of the image
        cv2.putText(image, txt, (30, 30), cv2.FONT_ITALIC, 0.6, (255, 0, 0), 1)


def detect_fishbody(thin_binary, image):
    """given binary image, proprose possible fishbody boudingboxes
    :param thin_binary: foreground skeleton image
    :param offset: compared with original image, origin offset of thin_binary
    :param image: original image, from which fish patches are cropped
    :return: boundingboxes that represent fishbody locations
    """
    # find connected regions
    num_labels, labels = cv2.connectedComponents(thin_binary, connectivity=8)
    # fit models using multi_seq_ransac algorithm
    ss = time()
    models = []
    for i in range(num_labels - 1):
        ys, xs = np.where(labels == i + 1)
        points = np.array((xs, ys)).T
        if points.shape[0] > min_score:
            is_easy_mode = points.shape[0] < easy_pattern_num
            group_models = multi_seq_ransac(points, is_easy_mode, min_score=min_score)  # , img_shape=thin_binary.shape)
            if len(group_models) > 0:
                models.append(group_models)
    print('--- ransac taking time: {:.2f}'.format(time() - ss))
    # improve models by extending fitted lines along underlying fishes' binary images
    models = RotatedRect.extend_lines(models, ext_percent=extention_percent)
    # form a rotated rectangle that encloses the fitted line with a  specified width
    rects = RotatedRect.compute_rects(models, image.shape[0:2])
    # crop out patches from frame image according to rects
    patches = RotatedRect.crop_rects(image, rects)
    return rects, patches


def detect_one_image(img_back, img_fore, image, x, y, w, h):
    start = time()
    # subtrack background and add foreground
    img = cv2.subtract(image, img_back, dtype=cv2.CV_32FC3)
    img = cv2.add(img, img_fore, dtype=cv2.CV_8UC3)
    img = pad_frame_image(img[y:y + h, x:x + w], pad_size=50)
    thin_img = thinning(img, foreground_bias=5)
    if np.sum(thin_img == 255) > 4 * 20 * (min_fish_length * 4):
        print("Exception of over-existence of foreground.")
        return [], []
    # fish detection and return rectangles enclosing fish, and cropped image patches
    cropped_image = pad_frame_image(image[y:y + h, x:x + w], pad_size=50)
    rects, patches = detect_fishbody(thin_img, image=cropped_image)
    # the offset w.r.t the original image coordinate system
    offset = np.array([x, y]) - np.array([50, 50])
    rects = RotatedRect.shift_rects(rects, offset=offset)
    # RotatedRect.draw_rects(image, rects)  # ' rect num: ' + str(rectnum)
    print('--detection one frame taking time: {}'.format(time() - start))
    return rects, patches


def test_detection_images():
    img_folder_path = r'D:\Fishes\images\imagesample_fishbody'
    folder_name = os.path.split(img_folder_path)[-1]
    backgrd_file = 'data\\foreback_grp\\img_backgrd_{}.jpg'.format(folder_name)
    foregrd_file = 'data\\foreback_grp\\img_foregrd_{}.jpg'.format(folder_name)
    if os.path.exists(backgrd_file) & os.path.exists(foregrd_file):
        img_fore = cv2.imread(foregrd_file)
        img_back = cv2.imread(backgrd_file)
    else:
        img_fore, img_back = compute_forebackground(img_folder_path,
                                                    backgrd_file=backgrd_file,
                                                    foregrd_file=foregrd_file)
    x, y, w, h = select_roi(img_back.astype(np.uint8))
    file_paths = []
    for cur_root, subdirs, files in os.walk(img_folder_path):
        file_paths.extend([os.path.join(cur_root, file) for file in files if file.endswith('.jpg')])
    for i in np.arange(0, len(file_paths), step=10):
        start = time()
        image = cv2.imread(file_paths[i])
        rects, _, = detect_one_image(img_back, img_fore, image, x, y, w, h)
        print('Detection on image {} taking: {}'.format(file_paths[i].split('\\')[-1], time() - start))
        cv2.namedWindow('detection', 0)
        cv2.imshow('detection', image)
        cv2.waitKey(0)


def test_detection_video():
    # video_fish_file = r'D:\Fishes\videos\2017_11_12\06.mp4'
    video_fish_file = r'D:\Fishes\videos\2018_03_22\01.wmv'
    video_name = video_fish_file.split('\\')[-2] + '_' + video_fish_file.split('\\')[-1].split('.')[0]
    backgrd_file = 'data\\foreback_grp\\img_backgrd_{}.jpg'.format(video_name)
    foregrd_file = 'data\\foreback_grp\\img_foregrd_{}.jpg'.format(video_name)
    if os.path.exists(backgrd_file) & os.path.exists(foregrd_file):
        img_fore = cv2.imread(foregrd_file)
        img_back = cv2.imread(backgrd_file)
    else:
        img_fore, img_back = compute_forebackground_video(video_fish_file,
                                                          backgrd_file=backgrd_file,
                                                          foregrd_file=foregrd_file)
    x, y, w, h = select_roi(img_back.astype(np.uint8))
    cap = cv2.VideoCapture(video_fish_file)
    num_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    for i in np.arange(0, num_frame, 30, dtype=int):
        start = time()
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        image = cap.read()[1]
        # fish detection and return rectangles enclosing fish, and cropped image patches
        rects, _ = detect_one_image(img_back, img_fore, image, x, y, w, h)
        print('Detection on frame {} taking: '.format(i, time() - start))
        cv2.namedWindow('detection', 0)
        cv2.imshow('detection', image)
        cv2.waitKey(0)


if __name__ == '__main__':
    # test_detection_images()
    test_detection_video()
