import numpy as np
import cv2, os
import pickle
from image_preprocess import *
from fishbody_detection import detect_one_image
from fishbody_recognition import FishClassifier, Fish
from fish_tracking import FishTracker, fish_num_to_track
from matplotlib import pyplot as plt


is_relearn_params = False
analysis_interval = 60
dead_score_thresh = 0.40
living_status = {0: 'dead', 1: 'normal', 2: 'struggling'}
param_file_path = 'data\\normal_behavior_param_20180503_01.pkl'
# when these two parameters are equal to 1, the mfs become triangular
# when these two parameters are equal to 0, the mfs become rectangular
left_vertice_offset_ratio = 0.8
right_vertice_offset_ratio = 0.8
# confidence level in extreme points
left_extreme_confidence_level = 0.30
right_extreme_confidence_level = 0.30
# overlap rate of two adjacent mfs
slow_normal_overlap_rate = 0.4
normal_fast_overlap_rate = 0.4


class IndividualBehaviorParam(object):
    def __init__(self):
        self.speed_median = np.zeros((3,))  # median of fish moving speed [vx, vy, w]
        self.acceleration_median = np.zeros((3,))  # median of fish moving acceleration [ax, ay, aw]
        self.distance_mean = 0  # the average moving distance of fish distance w.r.t its moving center
        self.curvature_std = 0  # the std of fish body curvature
        self.distribution_ratio = 0  # the distribution ratio = effective fish moving area / total image area

    def initialize(self, *args):
        if len(args) != 5:
            raise Exception('The number of args should be 5, but receive {}'.format(len(args)))
        self.speed_median = args[0]
        self.acceleration_median = args[1]
        self.distance_mean = args[2]
        self.curvature_std = args[3]
        self.distribution_ratio = args[4]

    def vectorize(self):
        """ extract values of each attribute and form a 1d array (a vector) """
        vector = np.hstack([self.speed_median, self.acceleration_median,
                            self.distance_mean, self.curvature_std, self.distribution_ratio])
        return vector

    def save_param(self, filename):
        with open(filename, 'wb') as f:  # Overwrites any existing file
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def read_param(filename):
        with open(filename, 'rb') as f:
            param = pickle.load(f)
        return param

    def __str__(self):
        sb = []
        for key in self.__dict__:
            sb.append("{key}='{value}'".format(key=key, value=self.__dict__[key]))
        return '\n'.join(sb)

    def __repr__(self):
        return self.__str__()


class NormalBehaviorParam(IndividualBehaviorParam):
    def __init__(self):
        # for each param, making statistics for its min, median, and max values
        super(NormalBehaviorParam, self).__init__()
        self.speed_extreme = np.zeros((3, 2))    # [min, max] of speed medians of a group of fishes
        self.acceleration_extreme = np.zeros((3, 2))  # [min, max] of acceleration medians of a group of fishes
        self.distance_extreme = np.array([0, 0])  # [min, max] of distance means of a group of fishes
        self.curvature_extreme = np.array([0, 0])  # [min, max] of curvature stds of a group of fishes
        self.distribution_extreme = np.array([0, 0])  # [min, max] of distribution ratios of a group of fishes

    def initialize(self, *args):
        if len(args) != 10:
            raise Exception('The number of args should be 10, but receive {}'.format(len(args)))
        super(NormalBehaviorParam, self).initialize(*args[0:5])
        self.speed_extreme = args[5]
        self.acceleration_extreme = args[6]
        self.distance_extreme = args[7]
        self.curvature_extreme = args[8]
        self.distribution_extreme = args[9]

    def vectorize(self):
        """ extract values of some attributes and then form a 1d array"""
        vector_median = super(NormalBehaviorParam, self).vectorize()
        vector_extreme = np.vstack([self.speed_extreme, self.acceleration_extreme,
                                    self.distance_extreme.reshape(1, 2),
                                    self.curvature_extreme.reshape(1, 2),
                                    self.distribution_extreme.reshape(1, 2)])
        return vector_median, vector_extreme

    def visualize_mfs(self):
        fig, axes = plt.subplots(3, 3)
        fuzzy_inferer = FuzzyInference(self)
        medians, extremes = self.vectorize()
        print(np.round(np.stack([extremes[:, 0], medians, extremes[:, 1]]), 2).T)
        X = np.empty((200, 9))
        for i in range(9):
            X[:, i] = np.linspace(-1, extremes[i, 1] * 2, 200)
        degrees = np.empty((200, 9, 3))
        for i in range(X.shape[0]):
            slows = fuzzy_inferer.mf_slow(X[i])
            normals = fuzzy_inferer.mf_normal(X[i])
            fasts = fuzzy_inferer.mf_fast(X[i])
            degrees[i, :, 0] = slows
            degrees[i, :, 1] = normals
            degrees[i, :, 2] = fasts

        for i in range(9):
            axe = axes[int(i // 3), int(i % 3)]
            axe.plot(X[:, i], degrees[:, i, 0], 'r')
            axe.plot(X[:, i], degrees[:, i, 1], 'b')
            axe.plot(X[:, i], degrees[:, i, 2], 'r')
            axe.vlines(medians[i], 0, 1, colors='k', linestyles='dashed')
            axe.vlines(extremes[i, 0], 0, 1, colors='y', linestyles='dashed')
            axe.vlines(extremes[i, 1], 0, 1, colors='y', linestyles='dashed')
        plt.tight_layout()
        plt.show()


class FuzzyInference(object):
    def __init__(self, normal_behavior_params, weights=None):
        """ infer a fish's moving or living status, slow(dead), normal, fast(struggling)
        :param normal_behavior_params: noraml behavior params
        :param weights: weights for each of behaviors of interest
        """
        self.medians, self.extremes = normal_behavior_params.vectorize()
        self.weights = weights  # an 1d array of size (9,), and summation equal to 1
        if weights is None:
            self.weights = np.array([0.1] * 3 + [0.1] * 3 + [0.25] + [0.10] + [0.05])

    def infer(self, behavior_param):
        """ According to fish behavior, infer its living status
        :param behavior_param: an instance of IndividualBehaviorParam class
        :return: inference result, 0--slow, 1--normal, 2--fast
        """
        X = behavior_param.vectorize()
        degrees = np.stack([self.mf_slow(X), self.mf_normal(X), self.mf_fast(X)]).T
        results = np.dot(self.weights, degrees)
        ind, score = np.argmax(results), np.max(results)
        ind = 1 if ind == 0 and score < dead_score_thresh else int(ind)
        print('-- current status: "{}" with status score {}'.format(living_status[ind], round(score, 1)))
        return ind

    def mf_normal(self, X):
        """ membership function for 'normal' linguistic value, which compute compatibility
            degree between given behavior X and normal behavior distribution
        :param X: given behavior params, an 1d array
        :return: compatibility degrees, which is winthin [0, 1]
        """
        def trapezoid(x, median, xmin, xmax):
            trapezoid_param = FuzzyInference.compute_trapezoid_param(median, xmin, xmax)
            left_vertice, left_slope, right_vertice, right_slope = trapezoid_param
            if left_vertice <= x <= right_vertice:
                return 1
            elif x < left_vertice:
                return max([0, left_slope * (x - left_vertice) + 1])
            else:
                return max([0, right_slope * (x - right_vertice) + 1])
        degrees = [trapezoid(x, self.medians[i], *self.extremes[i]) for i, x in enumerate(X)]
        return np.array(degrees)

    def mf_slow(self, X):
        """ membership function 'slow' linguistic value, which compute compatibility
            degree between given behavior X and slow behavior distribution
        """
        def trapezoid(x, median, xmin, xmax):
            left_vertice, left_slope, _, _ = FuzzyInference.compute_trapezoid_param(median, xmin, xmax)
            left_overlap_x = 1 / left_slope * (slow_normal_overlap_rate - 1) + left_vertice
            degree = - left_slope * (x - left_overlap_x) + slow_normal_overlap_rate
            return np.clip(degree, 0, 1)
        degrees = [trapezoid(x, self.medians[i], *self.extremes[i]) for i, x in enumerate(X)]
        return np.array(degrees)

    def mf_fast(self, X):
        """ membership function 'fast' linguistic value, which compute compatibility
                    degree between given behavior X and fast behavior distribution
        """
        def trapezoid(x, median, xmin, xmax):
            _, _, right_vertice, right_slope = FuzzyInference.compute_trapezoid_param(median, xmin, xmax)
            right_overlap_x = 1 / right_slope * (normal_fast_overlap_rate - 1) + right_vertice
            degree = - right_slope * (x - right_overlap_x) + normal_fast_overlap_rate
            return np.clip(degree, 0, 1)
        degrees = [trapezoid(x, self.medians[i], *self.extremes[i]) for i, x in enumerate(X)]
        return np.array(degrees)

    @staticmethod
    def compute_trapezoid_param(median, xmin, xmax):
        left_vertice = xmin + (median - xmin) * left_vertice_offset_ratio
        left_slope = (1 - left_extreme_confidence_level) / (left_vertice - xmin + 1e-3)
        right_vertice = xmax - (xmax - median) * right_vertice_offset_ratio
        right_slope = (right_extreme_confidence_level - 1) / (xmax - right_vertice + 1e-3)
        return left_vertice, left_slope, right_vertice, right_slope


class BehaviorAnalyzer(object):
    """  This class only focus on individual fishes, and gives information about fish mortality,
         by analyzing the behavior of a tracked fish once every short period (e.g., half a minute)
        and indicating whether the fish is live or dead within this period. If a fish is diagnosed
        as dead in two consecutive periods, then it'll be declared dead.
    """

    def __init__(self, analysis_interval, effective_image_size, param_file_path=None):
        """
        :param analysis_interval: analyziz fish behaviors per interval (frames)
        :param effective_image_size: the effective area where a fish can swim
        :param param_file_path: to or from which normal behavior params could be saved or loaded
        """
        self.analysis_interval = analysis_interval
        self.image_size = effective_image_size
        self.param_file_path = param_file_path
        self.normal_behavior_params = None
        self.fuzzy_inferer = None

    def analyze(self, traces):
        """
        Taking the normal behavior of group fishes as benchmark, analyze the behavior status of a given fish,
        and give a result, dead or live, to each tracked fish in traces
        :return: analysis results
        """
        results = [True] * len(traces)
        if self.fuzzy_inferer is None:
            raise Exception('Normal Fish Behavior Params have not been learned.')

        for i, trace in enumerate(traces):
            if len(trace.nodes) < self.analysis_interval:
                continue
            nodes = trace.nodes[-self.analysis_interval:]
            behavior_params = self.compute_behavior_params(nodes)
            status = self.fuzzy_inferer.infer(behavior_params)
            # moving normally or fast is considered to be a living fish,
            # moving slowly is considered to be a dead fish
            islive = True if status == 1 or status == 2 else False
            results[i] = islive
        return results

    def obtain_normal_behavior(self, traces, valid_ratio=0.75):
        """ Learning normal behavior parameters from all tracked fishes. So at the begining stage of an experiment,
            it is a must to make sure that all fishses are live and moving relatively normally.
        """
        def learn_params_online():
            # learn behavoirs from traces whose length is larger than analysis_interval
            valid_traces = [trace for trace in traces if len(trace.nodes) >= self.analysis_interval]
            if len(valid_traces) < len(traces) * valid_ratio:
                return None
            nodes_group = []
            for trace in valid_traces:
                length = len(trace.nodes)
                num_grp = min([5, length // self.analysis_interval])
                for i in range(0, num_grp):
                    nodes = trace.nodes[length - (i+1) * self.analysis_interval:length - i * self.analysis_interval]
                    nodes_group.append(nodes)
            # compute behavior params for each trace
            behaviors = [self.compute_behavior_params(nodes) for nodes in nodes_group]
            # obtain typical behavior characteristics by averaging group behaviors
            speed_median = np.stack([behavior.speed_median for behavior in behaviors])
            acceleration_median = np.stack([behavior.acceleration_median for behavior in behaviors])
            distance_mean = np.array([behavior.distance_mean for behavior in behaviors])
            curvature_std = np.array([behavior.curvature_std for behavior in behaviors])
            distribution_ratio = np.array([behavior.distribution_ratio for behavior in behaviors])

            normal_behavior = NormalBehaviorParam()
            normal_behavior.initialize(np.median(speed_median, axis=0),
                                       np.median(acceleration_median, axis=0),
                                       np.median(distance_mean),
                                       np.median(curvature_std),
                                       np.median(distribution_ratio),
                                       np.stack([np.min(speed_median, axis=0), np.max(speed_median, axis=0)]).T,
                                       np.stack([np.min(acceleration_median, axis=0), np.max(acceleration_median, axis=0)]).T,
                                       np.array([np.min(distance_mean), np.max(distance_mean)]),
                                       np.array([np.min(curvature_std), np.max(curvature_std)]),
                                       np.array([np.min(distribution_ratio), np.max(distribution_ratio)]))
            return normal_behavior

        if is_relearn_params is False and self.param_file_path is not None:
            if os.path.exists(self.param_file_path):
                normal_behavior = NormalBehaviorParam.read_param(self.param_file_path)
                print('--------------------------------------------------------------------------------')
                print('Loading fish normal behavior params from disk has finished.')
                print(normal_behavior)
                print('--------------------------------------------------------------------------------')
            else:
                raise Exception('The specified fish behavior param file does not exist.')
        else:
            normal_behavior = learn_params_online()
            if normal_behavior is not None:
                normal_behavior.save_param(self.param_file_path)
                print('--------------------------------------------------------------------------------')
                print('Learning fish normal behavior params online has finished.')
                print(normal_behavior)
                print('--------------------------------------------------------------------------------')
            else:
                print('--------------------------------------------------------------------------------')
                print('Fail to learn fish behavior params this time. The effective traces are not enough.')
                print('--------------------------------------------------------------------------------')

        if normal_behavior is not None:
            self.normal_behavior_params = normal_behavior
            self.fuzzy_inferer = FuzzyInference(self.normal_behavior_params)
            return True
        else:
            return False

    def compute_behavior_params(self, nodes):
        params = IndividualBehaviorParam()
        # obtain moving information
        head_locs = np.stack([node.fish.head_loc for node in nodes])
        orients = np.asarray([node.fish.orientation for node in nodes])
        curvatures = np.array([node.fish.rect.curvature for node in nodes])
        # compute speed params
        X = np.concatenate((head_locs, orients[:, None]), axis=1)
        params.speed_median = BehaviorAnalyzer.calculate_speed_median(X)
        # compute acceleration params
        params.acceleration_median = BehaviorAnalyzer.calculate_acceleration_median(X)
        # compute distance mean
        params.distance_mean = BehaviorAnalyzer.calculate_distance_mean(head_locs)
        # compute curvature params
        params.curvature_std = BehaviorAnalyzer.calculate_curvature_std(curvatures)
        # compute distribution ratio
        params.distribution_ratio = BehaviorAnalyzer.calculate_distribution_ratio(head_locs, self.image_size)
        return params

    @staticmethod
    def calculate_speed_median(X):
        """ compute the [median] of speed of a tracked fish
            since speed has signs, when computing it, we use its absolute values
        :param X: [x, y, orientation]
        :return: an array of size [3,] representing speed statistical parameters
        """
        speed = np.abs(np.diff(X, n=1, axis=0))
        median = np.median(speed, axis=0)
        return median

    @staticmethod
    def calculate_acceleration_median(X):
        """ compute the [median] of acceleration of speed of a tracked fish
            since acceleration has signs, when computing median we use its absolute values
        :param X: [x, y, orientation]
        :return: an array of size [3,] representing acceleration statistical parameters
        """
        acceleration = np.abs(np.diff(X, n=2, axis=0))
        median = np.median(np.abs(acceleration), axis=0)
        return median

    @staticmethod
    def calculate_distance_mean(X):
        """ compute the average displacement of a tracked fish w.r.t its moving center
        :param X: [x, y]
        :return: a scaler representing the average moving distance
        """
        moving_center = np.mean(X, axis=0)
        mean_distance = np.mean(np.linalg.norm(X - moving_center, axis=1))
        return mean_distance

    @staticmethod
    def calculate_curvature_std(X):
        """ compute the [std] of body curvature of a tracked fish
        :param X: [curvature]
        :return: a scalar representing curvature statistical parameter
        """
        return np.std(X)

    @staticmethod
    def calculate_distribution_ratio(X, image_size):
        """ compute the distribution of a tracked fish. The more regions the fish travels to,
            the higher distribution ratio is assigned.

            we assume that the stochastic movement of fish meet bivariate Gaussian Distribution.
            Each location in the trajecotry represent a 2d point in 2d space.

           1 unit_axis_length corresponds to 68% points within the error ellipse while 2 unit_axis_length
           corresponds to about 97.5% points. Then we compute the area of the region where a fish moves in 97.5% cases

        :param X: [x, y]
        :param image_size: (width, height)
        :return: a scaler representing the ratio to which fish moving region covers the total area
        """
        sigma = np.cov(X, rowvar=False)
        # eigenvalues relates the length of short and long axis of error ellipse
        eigen_values, _ = np.linalg.eigh(sigma)
        short_axis_length, long_axis_length = np.sqrt(eigen_values) * 2.0
        ellipse_area = np.pi * short_axis_length * long_axis_length
        ratio = ellipse_area / (image_size[0] * image_size[1])
        return ratio


def test_analyzer_video():
    video_name = '01_dead'
    video_fish_file = r'D:\Fishes\videos\\2018_05_03\{}.mp4'.format(video_name)
    grd_file_name = video_fish_file.split('\\')[-2]  # + '_' + video_fish_file.split('\\')[-1].split('.')[0]
    backgrd_file = 'data\\foreback_grp\\img_backgrd_{}.jpg'.format(grd_file_name)
    foregrd_file = 'data\\foreback_grp\\img_foregrd_{}.jpg'.format(grd_file_name)
    if os.path.exists(backgrd_file) & os.path.exists(foregrd_file):
        img_fore = cv2.imread(foregrd_file)
        img_back = cv2.imread(backgrd_file)
    else:
        img_fore, img_back = compute_forebackground_video(video_fish_file, image_num=1000,
                                                          backgrd_file=backgrd_file,
                                                          foregrd_file=foregrd_file)
    x, y, w, h = select_roi(img_back.astype(np.uint8))
    cap = cv2.VideoCapture(video_fish_file)
    num_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('fps: ', fps)
    # writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (1280, 720))
    fish_classifier = FishClassifier(fish_size=(80, 40))
    fish_tracker = FishTracker(fish_num=fish_num_to_track, moving_borders=np.array([x, y, w, h]))
    behavior_analyzer = BehaviorAnalyzer(analysis_interval, [w, h], param_file_path)

    is_train_stage = True
    for i in np.arange(900*1, num_frame, step=fps/5, dtype=int):
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
        # track fish
        if len(fishes) == 0:
            continue
        fish_tracker.track(fishes)
        # draw trajectories
        fish_tracker.draw_trajectory()

        if (is_relearn_params is False and is_train_stage) or \
                (is_train_stage and fish_tracker.currenr_frame_id % (analysis_interval * 5) == 0):
            is_obtained = behavior_analyzer.obtain_normal_behavior(fish_tracker.trace_manager.trace_pool)
            is_train_stage = False if is_obtained else True

        if is_train_stage is False and fish_tracker.currenr_frame_id % (analysis_interval / 10) == 0:
            fish_living_status = behavior_analyzer.analyze(fish_tracker.trace_manager.trace_pool)
            for status, trace in zip(fish_living_status, fish_tracker.trace_manager.trace_pool):
                trace.trace_status = status
            print("-- Status Identify Results: ", fish_living_status)
        # draw bounding boxes on image
        dic = {trace.tid: trace.trace_status for trace in fish_tracker.trace_manager.trace_pool}
        status = [True if fish.fid not in dic.keys() or dic[fish.fid] else False for fish in fishes]
        Fish.draw_fish_bboxes(image, fishes, status)

        num_rect = np.sum([1 for grp in rects for _ in grp])
        print('The process of fish  on frame id {1} taking time {0:.2f}'.format(time() - start, i))
        cv2.putText(image, str(num_rect) + '/' + str(len(fishes)), (20, 20),
                    cv2.FONT_ITALIC, 0.6, (255, 0, 0), 1)
        cv2.namedWindow('analyzing_{}'.format(video_name), 0)
        cv2.imshow('analyzing_{}'.format(video_name), image)
        # writer.write(image)
        cv2.waitKey(20)


if __name__ == '__main__':
    test_analyzer_video()
    # normal_behavior = NormalBehaviorParam.read_param(param_file_path)
    # normal_behavior.visualize_mfs()


