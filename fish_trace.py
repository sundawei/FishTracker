import numpy as np
import cv2
import os
import math
import time
from matplotlib import pyplot as plt
from kalam_filter import KalmanFilter


class TrackedFish(object):
    def __init__(self, fish, in_frame_id, in_trace_id=None):
        """ TrackedFish works as a node of FishTrace, wraping related information of a tracked fish
        """
        self.fish = fish  # data field, an instance of class Fish
        self.in_frame_id = in_frame_id  # the serial number of the frame image in which the fish exists
        self.in_trace_id = in_trace_id  # the ID of the trace into which the fish is classified
        self.fish.fid = self.fish.fid if in_trace_id is None else in_trace_id
        self.living_status = True  # True if fish is alive, False if dead

    def assign_trace_attributes(self, trace_id, trace_status):
        self.in_trace_id = trace_id
        self.fish.fid = trace_id
        self.living_status = trace_status


class FishTrace(object):
    """ A Trace Class Object represents a fish' moving trace across frames.
        It is instantiated only through FishTrackManager Class's create_fish_trace method.
    """
    def __init__(self, tid):
        self.tid = tid  # the ID of the trace, which should be kept unique.
        self.nodes = []  # the collection of trackedFishs in the trace
        self.skipped_frames = 0  # the number of frames the trace has been skipped continuously
        self.trace_status = True  # True if fish is alive, False if dead
        self.kalman_filter = KalmanFilter(t=1.0)

    moving_borders = np.array([0, 0, 0, 0])  # [x, y, w, h]

    def append(self, node):
        """ append a new detected object to the trace
        :param node: the new detected object as a node point of the trace
        """
        node.assign_trace_attributes(self.tid, self.trace_status)
        self.nodes.append(node)

        # initialize kalman filter using the first node's position
        if len(self.nodes) == 1:
            (x, y), theta = self.nodes[0].fish.head_loc, self.nodes[0].fish.orientation
            self.kalman_filter.init_state_vector(x=x, y=y, theta=theta)

    def compute_control_input(self):
        """ compute control input dynamically, being used before kalman_filter begins to predict
            use consecutive three frames' displacement differences to compute accelerations
        :return: control input vector
        """
        if len(self.nodes) < 3:
            return np.zeros(shape=(3, 1))
        # compute input control vector based on the lastest three positions
        head_locs = [node.fish.head_loc for node in self.nodes[-3:]]
        a_x, a_y = head_locs[2] + head_locs[0] - 2 * head_locs[1]
        # orientation within -180 and 180
        orients = [node.fish.orientation for node in self.nodes[-3:]]
        orients = [orient if orient >= 0 else orient + 360 for orient in orients]
        a_w = (orients[2] + orients[0] - 2 * orients[1]) % 360
        control_input = np.array([a_x, a_y, a_w]).reshape(3, 1)
        # for keeping system stability, clip control input vector
        control_input = np.clip(control_input, a_min=-20, a_max=20)
        return control_input

    def predict_current_position(self):
        """ predict object's current position based on object motion model
            as a priori position of the object.
        :return: predicted position
        """
        if len(self.nodes) == 0:
            raise Exception('No object has been tracked and existing in this trace.')

        # control_input = self.compute_control_input()
        # # if the tracked object got lost in previous frames, set control input vector to zero
        # if self.skipped_frames > 0:
        #     control_input = control_input * 0.0
        # else:
        #     control_input = control_input * 0.5  # decay control input vector to some extent

        # once having missed tracking, decay the speed -- special case 1
        if self.skipped_frames >= 1:
            self.kalman_filter.adjust_speed_state(decay_factor=0.5)
        # onece reaching one of the borders of the image, decay the speed -- special case 2
        cur_x, cur_y = self.kalman_filter.X[0:2, 0]
        close_h, close_v = FishTrace.check_borders(cur_x, cur_y)
        decay_factor = np.asarray([1 - close_h, 1 - close_v, 1]).reshape(3, 1)
        self.kalman_filter.adjust_speed_state(decay_factor=decay_factor)

        # predict a priori state vector X as well as corresponding priori covariance matrix
        self.kalman_filter.predict(u=None)  # control_input
        pred_position = self.kalman_filter.get_estimate_position()
        return pred_position

    def correct_current_position(self):
        """ correct the predicted position of the object by incorporating its measured position.
            This method should be called after predict_current_position which generates a predicted position,
            and append method which appends the best match to the trace, which gives a measured position.
            The measured_position is exactly the last node's fish attribute:[fish_loc[0], fish_loc[1], orient]
        """
        if len(self.nodes) < 2:
            raise Exception('No object has been tracked or only one existing in this trace.')
        # by combining measurement, correct predicted state vector and its covariance matrix,
        # and get a posteriori state vector and a posteriori covariance matrix.
        fish_loc, fish_orient = self.nodes[-1].fish.head_loc, self.nodes[-1].fish.orientation
        self.kalman_filter.correct(Z=np.asarray([fish_loc[0], fish_loc[1], fish_orient]).reshape(3, 1))
        corrected_position = self.kalman_filter.get_estimate_position()
        self.nodes[-1].fish.head_loc = corrected_position[0:2, 0]
        self.nodes[-1].fish.orientation = corrected_position[2, 0]
        return corrected_position

    @staticmethod
    def check_borders(cur_x, cur_y, thresh_distance=20):
        """ check whether current location is nearby one of the borders
        :param cur_x: current x coordinate
        :param cur_y: current y coordinate
        :param thresh_distance: the distance below which a fish is considered being close to a border
        :return: 1 if approximating a border, 0 else.
        """
        if np.all(FishTrace.moving_borders == 0):
            raise Exception('Having not initialized effective moving borders.')
        close_horizontal, close_vertical = 0, 0
        minx, miny, w, h = FishTrace.moving_borders
        if np.any(np.abs([cur_x - minx, cur_x - minx - w]) < thresh_distance):
            close_horizontal = 1
        if np.any(np.abs([cur_y - miny, cur_y - miny - h]) < thresh_distance):
            close_vertical = 1
        return close_horizontal, close_vertical


class FishTraceManager(object):
    """ FishTraceManager Class is responsible for creating a new FishTrace object,
        allocating a unique trace id to it, and then storing it.
        FishTraceManager used in Singleton Mode
    """
    __instance = None

    def __new__(cls, *args, **kwargs):
        if FishTraceManager.__instance is None:
            FishTraceManager.__instance = object.__new__(cls)
        return FishTraceManager.__instance

    def __init__(self, max_trace_num, max_trace_length, base_missing_frames, moving_borders):
        self.max_trace_num = max_trace_num  # the max number of traces allowed to create
        self.max_trace_length = max_trace_length  # the max number of nodes allowed in a trace
        self.base_missing_frames = base_missing_frames  # missing frames above which a trace may be dropped
        self.trace_pool = []  # FishTrace object pool where each element is a FishTrace object
        self.id_pool = []  # FishTrace ID pool where element is a trace id having been allocated.
        FishTrace.moving_borders = moving_borders  # the valid moving borders for fishes, i.e., ROI of image

    def create_fish_trace(self):
        """ create a new trace, allocate a unique link id to it
            and then add the created link to the linkpool
        :return: newly created trace
        """
        # if the number of traces has reached its max, no new trace is allowed to create
        if len(self.trace_pool) >= self.max_trace_num:
            return None

        # create a new trace and allocate an id for it
        tid = [i for i in range(len(self.id_pool) + 1) if i not in self.id_pool][0]
        trace = FishTrace(tid=tid)
        self.trace_pool.append(trace)
        self.id_pool.append(tid)
        return trace

    def drop_outdated_fish_traces(self):
        """ if a trace has not been updated exceeding a specified period, it will be dropped.
        :return: the number of traces being dropped
        """
        # threshold should be flexible, the longer trace the harder to drop
        def get_missing_frames_threshold(trace):
            times = np.clip([len(trace.nodes) / 100], 1, 4)
            return self.base_missing_frames * times

        indicator = [True if trace.skipped_frames < get_missing_frames_threshold(trace) else
                     False for trace in self.trace_pool]

        self.trace_pool = [trace for i, trace in enumerate(self.trace_pool) if indicator[i]]
        self.id_pool = [tid for i, tid in enumerate(self.id_pool) if indicator[i]]
        return np.sum(np.bitwise_not(indicator))

    def prune_overly_long_traces(self):
        """ prune overly longe traces to make not exceed max_trace_length
        """
        for trace in self.trace_pool:
            if len(trace.nodes) > self.max_trace_length:
                trace.nodes = trace.nodes[-self.max_trace_length:]
