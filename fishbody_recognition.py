import os
import cv2
import shutil
import glob
import numpy as np
from time import time
from keras.models import load_model
from shapely.geometry import Polygon
from image_preprocess import *
from fishbody_detection import detect_one_image, min_fish_length
from fish_modeling import fishhead_loss, hdacc

np.random.seed(1234)
fishnet_file = 'data\\model\\FISHNET-EP60-BODYACC0.9480-HEADACC0.9811.h5'
# threshold above which an image is seen as a fish
thresh_fish = 0.3
thresh_head = 0.3
# drop duplicate if (iom >= thresh_iom) & (iou >= thresh_iou)
thresh_iom = 0.7  # threshold value of intersection over min
thresh_iou = 0.6  # threhsold value of intersection over union

colors = [tuple(color) for color in np.random.randint(240, size=(25, 3)).tolist()] + [(0, 255, 0)]

# for fishid net
is_using_fishid = False
fishid_file = 'data\\model\\FISHID-EP142-ACC0.9465.h5'
fishid_mean_file = 'data\\fishid_mean.jpg'
thresh_fishid = 0.7


class Fish(object):
    def __init__(self, rect, score, orientation, group=-1, fid=-1):
        self.rect = rect  # the location of the fish
        self.score = score  # the score to be a fish, ∈ [0,1]
        self.orientation = orientation  # the angle between fishhead direction and positive x-axis, ∈[-180, 180)
        self.head_loc = np.array([0, 0])  # the location of fishhead
        self.group = group  # the No. of group the fish belongs to, ∈[1,20]
        self.fid = fid  # the identity of a fish (used afterwards)

    @staticmethod
    def form_fishes(rects, scores, ids):
        # transform the angle range of fish from [-90, 90) to [-180, 180)
        def transform_orien(origal_angle, head_score):
            if head_score < 0:
                return origal_angle + (180 if origal_angle < 0 else - 180)
            else:
                return origal_angle

        # determine the location of fish head
        def localize_fishhead(fish):
            if fish.orientation >= 0:
                return np.mean(fish.rect.corners[[1, 2]], axis=0)
            else:
                return np.mean(fish.rect.corners[[0, 3]], axis=0)

        cnt_grp = 1
        fishes = []
        angles = [rect.angle for grp in rects for rect in grp]
        oriens = [transform_orien(angle, head) for angle, head in zip(angles, scores[:, 1])]
        fishscores, fishids = scores[:, 0].tolist(), ids.tolist()
        isfishes = ((scores[:, 0] >= thresh_fish) & (np.abs(scores[:, 1]) >= thresh_head)).tolist()
        for i in range(len(rects)):
            group_fish = []
            for j in range(len(rects[i])):
                isfish, score, angle, fid = isfishes.pop(0), fishscores.pop(0), oriens.pop(0), fishids.pop(0)
                if isfish is True:
                    fish = Fish(rect=rects[i][j], score=score, orientation=angle, group=cnt_grp, fid=fid)
                    fish.head_loc = localize_fishhead(fish)
                    group_fish.append(fish)
            if len(group_fish) == 0:
                continue
            elif len(group_fish) >= 2:
                group_fish = Fish.drop_overlapped_fishes(group_fish, thresh_iom=thresh_iom, thresh_iou=thresh_iou)
            fishes.append(group_fish)
            cnt_grp += 1
        # unfold grouped fishes for convenient operations
        fishes = [fish for grp_fish in fishes for fish in grp_fish]
        return fishes

    @staticmethod
    def drop_overlapped_fishes(fishes, thresh_iom, thresh_iou):
        num = len(fishes)
        indicator = [True] * num
        # form polygons from rotated rects
        polygons = [Polygon([tuple(pt) for pt in fish.rect.corners]) for fish in fishes]
        # compute iou, iom etc.
        iom_arr = np.zeros((num, num))  # intersection area over min area of two polygons
        iou_arr = np.zeros_like(iom_arr)  # intersection over union area of two polygons
        for i in range(num - 1):
            for j in range(i + 1, num):
                overlap = polygons[i].intersection(polygons[j]).area
                iom_arr[i][j] = overlap / np.min([polygons[i].area, polygons[j].area])
                iou_arr[i][j] = overlap / polygons[i].union(polygons[j]).area

        cond1 = ((iom_arr >= thresh_iom) & (iou_arr >= thresh_iou)) | (iom_arr >= 0.8)
        # cond2 = (iom_arr >= 0.9) & (iom_arr < 1.0)
        rows, cols = np.where(cond1)
        for t in range(rows.shape[0]):
            if fishes[rows[t]].score <= fishes[cols[t]].score:
                indicator[rows[t]] = False
            else:
                indicator[cols[t]] = False
        fishes = [fishes[i] for i in range(num) if indicator[i] is True]
        return fishes

    @staticmethod
    def draw_fish_bboxes(image, fishes, status=None):
        rects = [fish.rect for fish in fishes]
        oriens = [fish.orientation for fish in fishes]
        scores = [fish.score for fish in fishes]
        ids = [fish.fid for fish in fishes]
        head_locs = [fish.head_loc for fish in fishes]
        for i in range(len(rects)):
            corners = [tuple(x) for x in rects[i].corners.astype(int).tolist()]
            center = tuple(rects[i].center.astype(int).tolist())
            # draw rectangles enclosing fish
            for j in range(len(corners)):
                cv2.line(image, corners[j - 1], corners[j],  colors[ids[i]], 2)
            if status is not None and status[i] is False:
                cv2.fillPoly(image, [rects[i].corners.astype(int)], (0, 0, 255))
            fishid_str = '_{:02d}'.format(ids[i]) if ids[i] != -1 else ''
            fishsc_str = str(round(scores[i], 1))
            cv2.putText(image, text=fishid_str, org=center, fontScale=0.8,
                        fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, color=(255, 0, 255), thickness=1)
            # highlight fish head direction, and draw two broken segments standing for curved fishbody
            anchors = [tuple(x) for x in rects[i].anchors.astype(int).tolist()]
            if oriens[i] >= 0:
                cv2.line(image, corners[1], corners[2], (0, 0, 255), 3)
                cv2.line(image, anchors[0], anchors[1], (255, 0, 0), 1)
                cv2.line(image, anchors[1], anchors[2], (0, 0, 255), 1)
            else:
                cv2.line(image, corners[0], corners[3], (0, 0, 255), 3)
                cv2.line(image, anchors[0], anchors[1], (0, 0, 255), 1)
                cv2.line(image, anchors[1], anchors[2], (255, 0, 0), 1)
            # localize fish head
            head_loc = tuple(head_locs[i].astype(int).tolist())
            cv2.circle(image, head_loc, 3, colors[ids[i]])


class FishClassifier(object):
    def __init__(self, fish_size=(80, 40)):
        # size of image patch accepted
        self.fish_size = fish_size  # (width, length)
        # load trained fishnet  model
        self.fishbody_model = load_model(fishnet_file, custom_objects=
                            {'fish_head_err': fishhead_loss(0), 'hdacc': hdacc})
        print("fishnet model is loaded from disk")
        if is_using_fishid:
            self.fishid_mean = cv2.imread(fishid_mean_file)
            self.fishid_model = load_model(fishid_file)
            print("fishid model is loaded from disk")

    def recognize_fishbody(self, patches):
        start = time()
        unfold_patches = [patch for grp in patches for patch in grp]
        unfold_patches = np.stack([cv2.resize(patch, dsize=self.fish_size) for patch in unfold_patches])
        norm_fishbody_patches = unfold_patches.astype(float) / 255
        fish_scores = self.fishbody_model.predict(norm_fishbody_patches)
        fish_scores = np.round(np.concatenate(fish_scores, axis=1), decimals=1)
        print('--- recogniting fish taking time: {:.2f}s'.format(time() - start))
        # ---------------------------------------------------------------------
        fish_ids = np.ones(fish_scores.shape[0], np.int32) * -1
        if is_using_fishid:
            fish_ids = self.recognize_fishid(unfold_patches, fish_scores)
        # -----------------------------------------------------------------------
        return fish_scores, fish_ids

    def recognize_fishbody2(self, processed_patches):
        start = time()
        fish_scores = self.fishbody_model.predict(processed_patches)
        fish_scores = np.round(np.concatenate(fish_scores, axis=1), decimals=1)
        print('--- recogniting fish taking time: {:.2f}s'.format(time() - start))
        # ---------------------------------------------------------------------
        fish_ids = np.ones(fish_scores.shape[0], np.int32) * -1
        # -----------------------------------------------------------------------
        return fish_scores, fish_ids

    def recognize_fishid(self, unfold_patches, fish_scores):
        norm_fishid_patches = (unfold_patches - self.fishid_mean.astype(float)) / 127
        id_scores = self.fishid_model.predict(norm_fishid_patches)
        fish_ids = np.argmax(id_scores, axis=1)
        fish_ids[id_scores[range(id_scores.shape[0]), fish_ids] < thresh_fishid] = -1
        fish_ids[fish_scores[:, 0] < thresh_fish] = -1
        return fish_ids


def test_recognition_video():
    video_name = '20mix'
    video_fish_file = r'D:\Fishes\videos\individual_fishes\20_20171228\20171228_mix\{}.mp4'.format(video_name)
    # video_fish_file = r'D:\Fishes\videos\2018_04_25\{}.mp4'.format(video_name)  # 2017_11_12
    grd_file_name = video_fish_file.split('\\')[-2] + '_' + video_fish_file.split('\\')[-1].split('.')[0]
    backgrd_file = 'data\\foreback_grp\\img_backgrd_{}.jpg'.format(grd_file_name)
    foregrd_file = 'data\\foreback_grp\\img_foregrd_{}.jpg'.format(grd_file_name)
    if os.path.exists(backgrd_file) & os.path.exists(foregrd_file):
        img_fore = cv2.imread(foregrd_file)
        img_back = cv2.imread(backgrd_file)
    else:
        img_fore, img_back = compute_forebackground_video(video_fish_file, image_num=500,
                                                          backgrd_file=backgrd_file,
                                                          foregrd_file=foregrd_file)
    x, y, w, h = select_roi(img_back.astype(np.uint8))
    cap = cv2.VideoCapture(video_fish_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fish_classifier = FishClassifier(fish_size=(80, 40))
    for i in np.arange(0, num_frame, step=fps/6, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        isvalid, image = cap.read()
        if isvalid is False:
            continue
        start = time()
        # fish detection and return rectangles enclosing fish, and cropped image patches
        rects, patches = detect_one_image(img_back, img_fore, image, x, y, w, h)
        if np.sum([1 for grp in patches for patch in grp]) == 0:
            continue
        # recognize patches
        scores, ids = fish_classifier.recognize_fishbody(patches)
        # form Fish class instances
        fishes = Fish.form_fishes(rects, scores, ids)

        # patches = [patch for grp in patches for patch in grp]
        # for j, [score, patch] in enumerate(zip(scores, patches)):
        #     head = 1 if score[1] >= 0 else 0
        #     if 0.3 <= score[0] < 0.95:
        #         file_name = "{}{}_{}.jpg".format(str(i).zfill(3), str(j).zfill(2), head)
        #         cv2.imwrite('data\\tmp\\' + file_name, patch)

        # draw boxes on image
        Fish.draw_fish_bboxes(image, fishes)
        num_rect = np.sum([1 for grp in rects for rect in grp])
        # num_fish = np.sum([1 for grp in fishes for fish in grp])
        print('The process of fish reconition on frame id {1} taking time {0:.2f}'.format(time() - start, i))
        cv2.putText(image, str(num_rect) + '/' + str(len(fishes)), (20, 20),
                    cv2.FONT_ITALIC, 0.6, (255, 0, 0), 1)
        cv2.namedWindow('recognition_{}'.format(video_name), 0)
        cv2.imshow('recognition_{}'.format(video_name), image)
        cv2.waitKey(100)


if __name__ == '__main__':
    test_recognition_video()
    # files = glob.glob('data\\tmp\\tmp2\\*.jpg')
    # heads = [os.path.split(file)[-1].split('_')[-1][0] for file in files]
    # patches = [cv2.imread(file) for file in files]
    # for i, patch in enumerate(patches):
    #     # flip horizontal and vertically
    #     fh = patch[:, ::-1]
    #     fv = patch[::-1, :]
    #     # resize and crop
    #     tmp = cv2.resize(patch, dsize=(int(patch.shape[1] * 1.2), int(patch.shape[0]*1.2)))
    #     tmp1 = tmp[-patch.shape[0]:, -patch.shape[1]:]
    #     tmp2 = tmp[:patch.shape[0], :patch.shape[1]]
    #
    #     cv2.imwrite('data\\tmp\\tmp2\\{}.jpg'.format(str(i+12040)), fh)
    #     cv2.imwrite('data\\tmp\\tmp2\\{}.jpg'.format(str(i + 12041)), fv)
    #     cv2.imwrite('data\\tmp\\tmp2\\{}.jpg'.format(str(i + 12042)), tmp1)
    #     cv2.imwrite('data\\tmp\\tmp2\\{}.jpg'.format(str(i + 12043)), tmp2)
