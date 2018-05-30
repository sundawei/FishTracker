import numpy as np
import cv2
import os
import math
import copy
from time import time
import scipy.optimize as optimizer
from matplotlib import pyplot as plt
from fish_trace import *
from image_preprocess import *
from fishbody_detection import detect_one_image
from fishbody_recognition import Fish, FishClassifier, colors
from fishid_descriptor import FishidDescriptor

fish_num_to_track = 21


class FishTracker(object):
    def __init__(self, fish_num, moving_borders):
        self.max_trace_num = fish_num  # the max number of traces allowed to create and exist
        self.max_trace_length = 2000  # the max number of frames buffered before start dropping oldest frames
        self.base_miss_peroid = 4  # the base missing tracking period allowed for a trace
        self.max_match_distance = 0.30  # the max allowed distance between two matched objects
        self.distance_weight = np.array([0.4, 0.4, 0.1])  # the distance weight of position vector [x, y, w]
        self.currenr_frame_id = 0  # record current frame id
        self.trace_manager = FishTraceManager(self.max_trace_num, self.max_trace_length,
                                              self.base_miss_peroid, moving_borders=moving_borders)

    def track(self, fishes):
        self.currenr_frame_id += 1
        if self.currenr_frame_id == 1:
            for fish in fishes:
                trace = self.trace_manager.create_fish_trace()
                if trace is not None:
                    trace.append(TrackedFish(fish, in_frame_id=self.currenr_frame_id))
                else:
                    print('The max number of trace has been reached.')
            return
        # prediction
        predicted_positions = [trace.predict_current_position() for trace in self.trace_manager.trace_pool]
        # matching
        detected_positions = [np.asarray([fish.head_loc.tolist() + [fish.orientation]]).T for fish in fishes]
        if len(predicted_positions) == 0 or len(detected_positions) == 0:
            return
        assignment = self.match_fishes(detected_positions, predicted_positions)
        # correction, if possible
        for i, trace in enumerate(self.trace_manager.trace_pool):
            if assignment[i] != -1:  # for matched fishes and traces, append the fish to the trace
                trace.append(TrackedFish(fishes[assignment[i]], in_frame_id=self.currenr_frame_id))
                trace.correct_current_position()
                trace.skipped_frames = 0
            else:  # a trace lossing its target, use predicted_position as the object's current location
                trace.skipped_frames += 1
                fish = copy.deepcopy(trace.nodes[-1].fish)
                fish.head_loc = predicted_positions[i][0:2, 0]
                fish.orientation = predicted_positions[i][2, 0]
                trace.append(TrackedFish(fish, in_frame_id=self.currenr_frame_id))
        # start new traces for unmatched fishes, if possible
        for i, fish in enumerate(fishes):
            if i not in assignment:
                trace = self.trace_manager.create_fish_trace()
                if trace is not None:
                    trace.append(TrackedFish(fish, in_frame_id=self.currenr_frame_id))
                else:
                    print('--- The max number of trace has been reached.')
        num_drop = self.trace_manager.drop_outdated_fish_traces()
        self.trace_manager.prune_overly_long_traces()
        if num_drop > 0:
            print('--- {} trace(s) have been dropped.'.format(num_drop))

    def match_fishes(self, detected_position, predicted_position):
        """ Given detected positions and predicted positions, match them to make the cost minimized
        :param detected_position: detected positions
        :param predicted_position: predicted positions
        :return:
        """
        predicted_pos = np.concatenate(predicted_position, axis=1).T  # nx3,  n traces existing
        detected_pos = np.concatenate(detected_position, axis=1).T  # mx3,  m fishes detected
        # calculate cost using distance between predicted and detected positions
        diff_pos = predicted_pos[:, None, :] - detected_pos[None, :, :]  # nxmx3
        per_channel_max_diff = np.max(np.max(np.abs(diff_pos), axis=0), axis=0)  # 3, max diff(x, y, theta)
        normalized_diff_pos = diff_pos / per_channel_max_diff  # nxmx3
        distance = np.sqrt(np.square(normalized_diff_pos).dot(self.distance_weight))  # nxm
        # use Hungarian Algorithm to assign the detected measurements (m jobs) to predicted traces (n workers)
        row_ind, col_ind = optimizer.linear_sum_assignment(distance)
        # job assignments for n workers, -1 stands for no job allocated to it
        assignment = np.ones(predicted_pos.shape[0], np.int32) * -1
        for r, c in zip(row_ind, col_ind):
            if distance[r, c] <= self.max_match_distance:
                assignment[r] = c
        # print('max distance: {:.2f}'.format(np.max(distance[row_ind, col_ind])))
        return assignment

    # __theta_fig, __axes = plt.subplots(1, 1)

    def draw_trajectory(self):
        # def init_theta_graph():
            # FishTracker.__axes.cla()
            # FishTracker.__axes.set_xlim([0, self.max_trace_length])  # the frame id
            # FishTracker.__axes.set_ylim([-180, 180])  # the rotation angle
            # FishTracker.__axes.set_xlabel('No. of frame')
            # FishTracker.__axes.set_ylabel('angle/degree')
            # FishTracker.__axes.set_autoscaley_on(False)

        graph = np.ones((720, 1280, 3), np.uint8) * 255
        for trace in self.trace_manager.trace_pool:
            if len(trace.nodes) < 2:
                continue
            xy = [tuple(node.fish.head_loc.astype(int).tolist()) for node in trace.nodes]

            # draw (x, y) position
            for i in range(1, len(xy)):
                cv2.line(graph, xy[i - 1], xy[i], colors[trace.tid], 1)

                # theta = np.asarray([node.fish.orientation for node in trace.nodes])
                # init_theta_graph()
                # color = tuple(x / 255.0 for x in colors[trace.tid])
                # FishTracker.__axes.plot(theta, '-', color=color)

        # plt.show(block=False)
        # plt.pause(0.01)
        cv2.namedWindow('Fish_2D_Motion', 0)
        cv2.imshow('Fish_2D_Motion', graph)
        # cv2.waitKey(10)


def test_tracking_video():
    video_name = '01'
    video_fish_file = r'D:\Fishes\videos\2018_04_25\{}.mp4'.format(video_name)  # 2018_04_25
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
    num_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    writer = cv2.VideoWriter('data\\output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (1280, 720))
    fish_classifier = FishClassifier(fish_size=(80, 40))
    fish_trackor = FishTracker(fish_num=fish_num_to_track, moving_borders=np.asarray([x, y, w, h]))

    for i in np.arange(0, num_frame, 3, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        isvalid, image = cap.read()
        if isvalid is False:
            continue
        start = time()
        # fish detection and return rectangles enclosing fish, and cropped image patches
        rects, patches = detect_one_image(img_back, img_fore, image, x, y, w, h)
        if np.sum([1 for grp in patches for patch in grp]) == 0:
            continue
        unfold_patches = [patch for grp in patches for patch in grp]
        unfold_patches = np.stack([cv2.resize(patch, dsize=(80, 40)) for patch in unfold_patches])
        scaled_patches = unfold_patches.astype(float) / 255
        # recognize patches
        scores, ids = fish_classifier.recognize_fishbody2(scaled_patches)
        # form Fish class instances
        fishes = Fish.form_fishes(rects, scores, ids)
        # track fish
        if len(fishes) == 0:
            continue
        fish_trackor.track(fishes)
        # draw trajectories
        fish_trackor.draw_trajectory()
        # draw bounding boxes on image
        Fish.draw_fish_bboxes(image, fishes)

        num_rect = np.sum([1 for grp in rects for _ in grp])
        print('The process of fish tracking on frame id {1} taking time {0:.2f}'.format(time() - start, i))
        cv2.putText(image, str(num_rect) + '/' + str(len(fishes)), (20, 20),
                    cv2.FONT_ITALIC, 0.6, (255, 0, 0), 1)
        cv2.namedWindow('tracking_{}'.format(video_name), 0)
        cv2.imshow('tracking_{}'.format(video_name), image)
        writer.write(image)
        cv2.waitKey(100)


if __name__ == '__main__':
    test_tracking_video()
