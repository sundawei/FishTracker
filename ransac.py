import math
import time
import cv2
import numpy as np

max_segment_gap = 5
cutoff_deviation = 9


def compute_distances(points):
    distances = points - np.mean(points, axis=0)
    signs = np.ones(distances.shape[0])
    signs[distances[:, 1] < 0] = -1
    signs[(distances[:, 1] == 0) & (distances[:, 0] < 0)] = -1
    distances = np.sqrt(np.sum(distances ** 2, axis=1)) * signs
    return distances


def find_range(distance, gap_threshold, weight=None):
    """ find a continuous segment which has max score (max number of points)
    :param distance: 1D array standing for coordinates along an axis
    :param gap_threshold: max gap allowed between two points / segments
    :param weight: the score of each point, by default, it's 1 for all points
           but it may decay as a point is occupied many times by several lines
    :return: the indices of points constituting the longest segment
    """
    # valid element indices of input array, distance
    I = np.argsort(distance)
    sorted_distance = distance[I]
    # find gap locations
    gapindex = np.where(np.diff(sorted_distance) > gap_threshold)[0]
    gapindex = np.hstack((0, gapindex, I.shape[0] - 1)).astype(np.int32)
    if gapindex.size == 2:
        startptr = gapindex[0]
        endptr = gapindex[-1]
    else:
        segment_score = np.zeros(gapindex.shape[0]-1)
        if weight is not None:
            for i in range(gapindex.shape[0]-1):
                segment_score[i] = np.sum(weight[I[gapindex[i]:gapindex[i+1]]])
        else:
            segment_score = np.diff(gapindex)
        ind = np.argmax(segment_score)
        startptr = gapindex[ind] + 1
        endptr = gapindex[ind + 1]
    return I[startptr:endptr+1]


def weighting(x, cutoff_deviation):
    """approximate a step function, the larger distance the smaller weight
    :param x: represente distance between a point and a line
    :param cutoff_deviation: the max point-to-line deviation distance, above which its weight is set to 0.
           for consensus points, each one has a weight depent on its distance to the consensus line.

           weight > 0.5  when distance < cutoff_deviation
           weight = 0.5  when distance == cutoff_deviation
           weight = 0    when distance > cutoff_deviation

    :return: the value (i.e., weight) assigend to each distance x
    """
    y = np.exp(x / cutoff_deviation * np.log(0.5))
    y[y < 0.5] = 0
    return y


def compute_linear_eqaution(endpt):
    """calculate model parameters of a straight line instantiated by two endpoints
    :param endpt: two endpoints which define a line
    :return: parameter k and b of equation y = k*x + b
    """
    delta_x, delta_y = endpt[1, 0] - endpt[0, 0], endpt[1, 1] - endpt[0, 1]
    if (delta_x != 0) & (np.abs(delta_y / (delta_x + 1e-3)) <= 50):
        k = delta_y / delta_x
        b = endpt[1, 1] - k * endpt[1, 0]
    else:
        k = -np.inf
        b = np.mean(endpt[:, 0])
    return k, b


def random_sample_concensus_v2(points, min_score, img_shape=None):
    """given a set of points, find several straight lines which fit points well
       compared with random_sample_concensus_v1, this algorithm is more general
       but consumes more time when running. So it is used for complicated mode.

       first,  define a number of lines by randomly selecting pairs of points
       second, find fitness-of-good lines from all defined lines, good fitness
               means a line has lots of consensus points
       third,  selecte lines recursively by dramamic programming with such a goal
               that few number of lines fit most points. Thus, selected lines shold
               ovelap with each other as less as possible.

    :param points: coordinates of given points, [x, y]
    :param min_score: min score a selected line shold get
    :param img_shape: only for visualization
    :return: models
    """
    model = {'endpts': [], 'anchors': []}
    # the number of pairs of points selected
    num_pairs = 100
    # selected two points defines a straight line
    frst, sec = np.random.randint(1, points.shape[0], size=(2, num_pairs))
    frst[frst == sec] = 0
    select_points = np.hstack((points[frst], points[sec])).T
    x1, y1, x2, y2 = select_points
    # projection coordinates of points into a line defined by two points
    o = np.array([x1, y1]).T  # m*2
    v1 = points[None, :, :] - o[:, None, :]  # 1*n*2 - m*1*2 = m*n*2
    v2 = np.array([x2 - x1, y2 - y1]).T  # m*2
    v1v2 = np.sum(v1 * v2[:, None, :], axis=2)  # sum(m*n*2 * m*1*2, axis=2) = m*n
    scale = v1v2[:, :, None] * v2[:, None, :]  # m*n*1 * m*1*2 = m*n*2
    length = np.sum(v2 ** 2, axis=1)[:, None, None]  # m*1*1
    # projection points and distances from original points to projection points
    projpts = scale / length + o[:, None, :]  # m*n*2 + m*1*2 = m*n*2
    distances = np.linalg.norm(projpts - points, axis=2)  # m*n
    values = weighting(distances, cutoff_deviation)  # m*n

    # look for a segment of longest length within a straight line
    endpts = np.zeros((num_pairs, 2, 2))
    anchorpts = np.zeros((num_pairs, 3, 2))
    labels = values > 0  # m*n
    for i in range(distances.shape[0]):
        inds = labels[i]
        projected = projpts[i][inds]
        # compute the adjacent distances from each projected point to their center
        adjacent_distances = compute_distances(projected)
        # find the longst continuous segment consisting of discrete points
        ranges = find_range(adjacent_distances, max_segment_gap)
        endpt = np.array([projected[ranges[0]], projected[ranges[-1]]])
        anchorpt = np.array([points[inds][ranges[0]], np.mean(endpt, axis=0), points[inds][ranges[-1]]])
        logical = np.zeros((points.shape[0],), np.bool)
        location = np.where(inds)[0][ranges]
        logical[location] = True
        labels[i] = logical
        values[i][np.bitwise_not(logical)] = 0
        endpts[i] = endpt
        anchorpts[i] = anchorpt

    # recursively select lines
    rows = []
    inliers = np.zeros((1, points.shape[0]), dtype=np.bool)  # k*n
    cover = inliers[0]
    while True:
        overlapmat = labels[:, None, :] & inliers  # m*k*n
        index = np.argmax(np.sum(overlapmat, axis=2), axis=1)  # m
        overlapmat = overlapmat[range(index.shape[0]), index]  # m*n
        overlaprate = np.sum(overlapmat, axis=1) / np.sum(labels, axis=1)
        decay = np.ones_like(overlapmat, dtype=np.float32)
        for i in range(decay.shape[0]):
            decay[i][overlapmat[i]] = np.exp(overlaprate[i]**2 / -0.25)
        score = np.sum(values * decay, axis=1)
        ind = np.argmax(score)
        rows.append(ind)
        inliers = np.vstack((inliers, labels[ind]))
        cover = cover | inliers[-1]

        if (np.sum(cover) > points.shape[0] - min_score*1.0) | (score[ind] < min_score):
            break

    models = []
    for i, r in enumerate(rows):
        # k, b = compute_linear_eqaution(endpts[r])
        # save model parameters
        model['endpts'] = endpts[r]
        model['anchors'] = anchorpts[r]
        models.append(model.copy())

    if img_shape is not None:
        img = np.ones(img_shape + (3,), np.uint8) * 255
        img[points[:, 1], points[:, 0]] = np.array([0, 0, 0])
        cv2.putText(img, 'num of pts: {}'.format(points.shape[0]), (20, 20), cv2.FONT_ITALIC, 0.5, (255, 0, 0), 1)
        for i, r in enumerate(rows):
            pts = [tuple(pt) for pt in endpts[r].astype(int).tolist()]
            cv2.line(img, pts[0], pts[1], (0, 0, 255), 1)
            img[points[inliers[i+1], 1], points[inliers[i+1], 0]] = np.array([255, 0, 0])
            cv2.namedWindow('img', 0)
            cv2.imshow('img', img)
            cv2.waitKey(0)
        cv2.destroyWindow('img')
    return models


def random_sample_concensus_v1(points, counters):
    """given a set of points, find several straight lines which fit points well
       compared with random_sample_concensus_v2, this algorithm is simplified and
       tricky, used for easy mode.
    :param points: coordinates of given points, [x, y]
    :param counters: record the number of times a point has been used totally
    :return: model
    """
    model = {'endpts': [], 'anchors': []}
    # the number of pairs of points selected
    num_pairs = 150
    rows = np.where(counters == 0)[0]
    # selected two points defines a straight line
    frst, sec = np.random.randint(1, rows.shape[0], size=(2, num_pairs))
    frst[frst == sec] = 0
    randompts = np.hstack((points[rows[frst]], points[rows[sec]])).T
    x1, y1, x2, y2 = randompts
    # distances from points to a line
    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    triangle_area = np.abs(np.outer(y2 - y1, points[:, 0]) -
                           np.outer(x2 - x1, points[:, 1]) +
                           (x2 * y1 - y2 * x1)[:, None])
    distances = triangle_area / length[:, None]
    values = weighting(distances, cutoff_deviation)
    # point decay
    values = values * np.exp(counters * -1)  # point_decay_factor
    # length decay
    labels = values > 0
    # num_points = np.sum(labels, axis=1)
    # selectedpts = points * labels[:, :, None]  # n*2 * m*n*2 = m*n*2
    # xymax = np.max(selectedpts, axis=1)  # m*2
    # selectedpts[selectedpts == 0] = 2**16
    # xymin = np.min(selectedpts, axis=1)  # m*2
    # line_length = np.linalg.norm(xymax - xymin, axis=1)
    # density = num_points / line_length
    # density[density > 1] = 1
    # scores = np.sum(values, axis=1) * density
    scores = np.sum(values, axis=1)
    # find the max score
    rowth = np.argmax(scores)
    # the two points defining the good-fitness line
    x1, y1, x2, y2 = randompts[:, rowth]
    # check which points are close to the straight line
    inds = labels[rowth]
    # projection points of these close points into the straight line
    v1 = points[inds] - np.array([x1, y1])
    v2 = np.array([x2-x1, y2-y1])
    projected = np.outer(v1.dot(v2), v2) / v2.dot(v2) + np.array([x1, y1])
    # compute the adjacent distances from each projected point to their center
    adjacent_distances = compute_distances(projected)
    # find the longst continuous segment consisting of discrete points
    ranges = find_range(adjacent_distances, gap_threshold=max_segment_gap, weight=np.exp(counters * -1)[inds])
    endpts = np.array([projected[ranges[0]], projected[ranges[-1]]])
    anchors = np.array([points[inds][ranges[0]], np.mean(endpts, axis=0), points[inds][ranges[-1]]])
    inliers = np.zeros((points.shape[0],), np.bool)
    locations = np.where(inds)[0][ranges]
    inliers[locations] = True

    # save model parameters
    # k, b = compute_linear_eqaution(endpts)
    model['endpts'] = endpts
    model['anchors'] = anchors

    return model, inliers, scores[rowth]


def multi_seq_ransac(points, is_easy_model, min_score, img_shape=None):

    if is_easy_model is False:
        models = random_sample_concensus_v2(points, min_score, img_shape)
        return models

    if img_shape is not None:
        img = np.ones(img_shape + (3,), dtype=np.uint8) * 255
        img[points[:, 1], points[:, 0]] = np.array([0, 0, 0])
        cv2.putText(img, 'num of pts: {}'.format(points.shape[0]), (20, 20), cv2.FONT_ITALIC, 0.5, (255, 0, 0), 1)

    models = []
    counters = np.zeros(points.shape[0], np.int32)
    recorders = np.zeros((1, points.shape[0]), np.bool)
    while np.sum(counters == 0) > (min_score*2/3):
        model, inliers, score = random_sample_concensus_v1(points, counters)
        counters = counters + inliers
        iou = np.sum(recorders & inliers, axis=1) / np.sum(recorders | inliers, axis=1)
        if (score < min_score) | (np.max(iou) > 0.9):
            break
        if np.all(iou <= 0.7):
            models.append(model)
            recorders = np.vstack((recorders, inliers))
            if img_shape is not None:
                pts = [tuple(pt) for pt in np.asarray(model['endpts'], dtype=np.int32).tolist()]
                img[points[inliers, 1], points[inliers, 0]] = np.array([255, 0, 0])
                cv2.line(img, pts[0], pts[1], (0, 0, 255), 1)
                cv2.namedWindow('fitline', 0)
                cv2.imshow('fitline', img)
                cv2.waitKey(0)
    return models


