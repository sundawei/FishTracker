import os
import cv2
import numpy as np
from time import time
from skimage.morphology import skeletonize, medial_axis
from skimage import img_as_ubyte, img_as_bool


patch_pad_v = cv2.imread('data\\padding\\patch_pad_v.png')
patch_pad_h = cv2.imread('data\\padding\\patch_pad_h.png')
image_pad_v = cv2.imread('data\\padding\\image_pad_v.png')
image_pad_h = cv2.imread('data\\padding\\image_pad_h.png')


def read_image_from_video(video_file, image_num):
    cap = cv2.VideoCapture(video_file)
    if cap.isOpened() is False:
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    totalnum = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('fps: {}, frames {}'.format(fps, totalnum))
    _, img = cap.read()
    data = np.zeros((image_num,) + img.shape, dtype=np.uint8)
    for i, ind in enumerate(np.linspace(0, totalnum, image_num, dtype=int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, ind - 1)  # set frame position to read
        isvalid, img = cap.read()
        if isvalid:
            data[i] = img
            print('img {} -- frame id {}'.format(i, ind))
        else:
            break
    print('The number of image read from video: ', i+1)
    return data


def pad_frame_image(image, pad_size=50):
    """add horizontal and vertical paddings to given frame image
    :param image: given frame image
    :param pad_size: padding size from all four directions
    :return: padded image
    """
    global image_pad_h, image_pad_v

    # resize image patch used as horizontal padding pixels
    if image.shape[0] <= 0:
        print('image error in padding. image shape {}'.format(image.shape))

    if image_pad_h.shape[0] <= 0:
        print('patch error in padding.')
    padding = cv2.resize(image_pad_h, dsize=(pad_size, image.shape[0]))
    # pad the left and right side of the original image
    image = np.hstack((padding, image, padding))
    # resize image patch used as vertical padding pixels
    padding = cv2.resize(image_pad_v, dsize=(image.shape[1], pad_size))
    # pad the upper and lower side of the image
    image = np.vstack((padding, image, padding))
    # appy a filter to region of conjunction
    i,j = pad_size - 3, pad_size + 3
    image[i:j] = cv2.GaussianBlur(image[i:j], (5, 5), 5)
    image[-j:-i] = cv2.GaussianBlur(image[-j:-i], (5, 5), 5)
    image[:, i:j] = cv2.GaussianBlur(image[:, i:j], (5, 5), 5)
    image[:, -j:-i] = cv2.GaussianBlur(image[:, -j:-i], (5, 5), 5)
    return image


def pad_fish_patch(patch, pad_size=5):
    """ add vertical paddings to a fish patch
    :param patch: given fish patch
    :param pad_size: padding size
    :return: padded patch
    """
    # resize image patch used as vertical padding pixels
    padding = cv2.resize(patch_pad_v, dsize=(patch.shape[1], pad_size))
    # pad the upper and lower side of the image
    patch = np.vstack((padding, patch, padding))
    # appy a filter to region of conjunction
    i, j = pad_size - 3, pad_size + 2
    patch[i:j] = cv2.GaussianBlur(patch[i:j], (3, 3), 2)
    patch[-j:-i] = cv2.GaussianBlur(patch[-j:-i], (3, 3), 2)
    return patch


def thinning(image, foreground_bias=10):
    """given a RGB image, get its foreground skeleton
    :param image: 3-channel image
    :param foreground_bias: bias ostu theshold value to make sure all fishes are foreground
    :return: thinned image which is a binary image
    """
    start = time()
    # binarization
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (3, 3), 0)
    res, _ = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU
    res, binary = cv2.threshold(gray, res+foreground_bias, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_not(binary)
    # remove fish fins
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    erosion = cv2.erode(binary, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    # perform skeletonization or thinning
    bn = img_as_bool(dilation, force_copy=True)
    skeleton = skeletonize(bn)
    thin_img = img_as_ubyte(skeleton)
    cv2.imshow('thin_img', dilation)
    print('--- thinning taking time: {:.2f}'.format(time() - start))
    return thin_img




def compute_forebackground(root, image_num=1000,
                           backgrd_file='data\\img_backgrd.jpg',
                           foregrd_file='data\\img_foregrd.jpg'):
    # read all image files
    file_paths = []
    for cur_root, subdirs, files in os.walk(root):
        file_paths.extend([os.path.join(cur_root, file) for file in files if file.endswith('.jpg')])
    np.random.shuffle(file_paths)
    # select 1000 images fro computing background
    num = np.min([image_num, len(file_paths)])
    files = file_paths[0:num]
    img = cv2.imread(files[0])
    data = np.empty((num,)+img.shape, dtype=img.dtype)
    for i, file in enumerate(files):
        data[i] = cv2.imread(file)
        print("img {}".format(i))
    img_backgrd = np.mean(data, axis=0)
    # compute foreground
    value_foregrd = (np.mean(np.mean(image_pad_v, axis=0), axis=0) +
                   np.mean(np.mean(image_pad_h, axis=0), axis=0)) / 2
    img_foregrd = np.zeros_like(img) + value_foregrd
    cv2.imwrite(backgrd_file, img_backgrd)
    cv2.imwrite(foregrd_file, img_foregrd)
    return img_foregrd, img_backgrd


def compute_forebackground_video(file_path, image_num=600,
                                 backgrd_file='data\\img_backgrd.jpg',
                                 foregrd_file='data\\img_foregrd.jpg'):
    # compute foreground
    value_foregrd = (np.mean(np.mean(image_pad_v, axis=0), axis=0) +
                     np.mean(np.mean(image_pad_h, axis=0), axis=0)) / 2

    data = read_image_from_video(file_path, image_num=image_num)
    img_backgrd = np.asarray(np.sum(data, axis=0) / image_num, np.uint8)
    img_foregrd = np.asarray(np.zeros_like(img_backgrd) + value_foregrd, np.uint8)

    cv2.imwrite(backgrd_file, img_backgrd)
    cv2.imwrite(foregrd_file, img_foregrd)
    return img_foregrd, img_backgrd


def inexact_augmented_lagrange_multiplier(X, lmbda=.015, tol=1e-3, maxiter=30):
    """
    Inexact Augmented Lagrange Multiplier
    """
    Y = X
    norm_two = np.linalg.norm(Y.ravel(), 2)
    norm_inf = np.linalg.norm(Y.ravel(), np.inf) / lmbda
    dual_norm = np.max([norm_two, norm_inf])
    Y = Y / dual_norm
    A = np.zeros(Y.shape)
    E = np.zeros(Y.shape)
    dnorm = np.linalg.norm(X, 'fro')
    mu = 1.25 / norm_two
    rho = 1.5
    sv = 10.
    n = Y.shape[0]
    itr = 0
    while True:
        Eraw = X - A + (1 / mu) * Y
        Eupdate = np.maximum(Eraw - lmbda / mu, 0) + np.minimum(Eraw + lmbda / mu, 0)
        U, S, V = np.linalg.svd(X - Eupdate + (1 / mu) * Y, full_matrices=False)
        svp = (S > 1 / mu).shape[0]
        if svp < sv:
            sv = np.min([svp + 1, n])
        else:
            sv = np.min([svp + round(.05 * n), n])
        Aupdate = np.dot(np.dot(U[:, :svp], np.diag(S[:svp] - 1 / mu)), V[:svp, :])
        A = Aupdate  # A is low rank matrix
        E = Eupdate  # E is sparse matrix
        Z = X - A - E  # Z is constrain matrix
        Y = Y + mu * Z  # Y is Lagrange multiplier matrix
        mu = np.min([mu * rho, mu * 1e7])
        itr += 1

        err = np.linalg.norm(Z, 'fro') / dnorm
        if (err < tol) or (itr >= maxiter):
            break
        print("iteration {}, error {}".format(itr, err))
        # cv2.imshow('origin',  (X[:, 10].reshape(d1, d2)*255).astype(np.uint8))
        # cv2.imshow('LowRank', (A[:, 10].reshape(d1, d2)*255).astype(np.uint8))
        # cv2.imshow('Sparse',  (E[:, 10].reshape(d1, d2)*255).astype(np.uint8))
        # cv2.waitKey(100)
    return A, E


def background_modeling_rpca(video_file, image_num=500, zoom_ratio=4):
    images = read_image_from_video(video_file, image_num=image_num)
    num, h, w = images.shape[0:3]
    h, w = int(h/zoom_ratio), int(w/zoom_ratio)
    data = np.empty((num, h, w), dtype=np.uint8)
    for i in range(num):
        data[i] = cv2.cvtColor(cv2.resize(images[i], dsize=(w, h)), cv2.COLOR_BGR2GRAY)
    X = data.reshape(num, -1).T / 255.0  # scaling
    print('X shape: ', X.shape)
    print('Begin Solving Robust PCA...')
    A, E = inexact_augmented_lagrange_multiplier(X)
    print('Solving Robust PCA finished.')

    A = A.T.reshape(num, h, w) * 255
    E = E.T.reshape(num, h, w) * 255

    # # fishbody is dark color compared to fishtank
    # E[E > 0] = 0
    # E[E < -10] = 255
    # E = E.astype(np.uint8)
    # A = A.astype(np.uint8)
    # for i in range(data.shape[0]):
    #     cv2.imshow('Original', cv2.resize(data[i], dsize=images.shape[2:0:-1]))
    #     cv2.imshow('LowRank', cv2.resize(A[i], dsize=images.shape[2:0:-1]))
    #     cv2.imshow('Sparse', cv2.resize(E[i], dsize=images.shape[2:0:-1]))
    #     cv2.waitKey(0)
    return A, E


def select_roi(image):
    global origin_img, prept, curpt
    origin_img = image
    while True:
        cv2.namedWindow('buffer_img', 0)
        cv2.setMouseCallback('buffer_img', on_mouse)
        cv2.imshow('buffer_img', image)
        key = cv2.waitKey(0)
        # if key == 'r' or 'R', reselect roi
        if (key != 114) & (key != 82):
            break
    cv2.destroyWindow('buffer_img')
    pts = np.array([prept, curpt])
    x, y = np.min(pts, axis=0)
    w, h = np.abs(pts[0] - pts[1])
    print('ROI selected (x, y, w, h): ', x, y, w, h)
    return x, y, w, h


prept, curpt, origin_img = (0, 0), (0, 0), []


def on_mouse(event, x, y, flags, param):
    global prept, curpt, origin_img
    # as left button is down which means roi selection begins,
    # read coordinates and draw a circle at the start point
    if event == cv2.EVENT_LBUTTONDOWN:
        prept = (x, y)
        buffer_img = np.copy(origin_img)
        text = '({},{})'.format(x, y)
        cv2.circle(buffer_img, prept, 2, (0, 0, 255), 2)
        cv2.putText(buffer_img, text, prept, cv2.FONT_ITALIC, 0.3, (0, 255, 0), 1)
        cv2.imshow('buffer_img',buffer_img)
    # as left button is up which means roi selection ends,
    # read coordinates and draw a rectangle with two points
    elif event == cv2.EVENT_LBUTTONUP:
        curpt = (x, y)
        buffer_img = np.copy(origin_img)
        text = '({},{})'.format(x, y)
        cv2.circle(buffer_img, prept, 2, (0, 0, 255), 2)
        cv2.circle(buffer_img, curpt, 2, (0, 0, 255), 2)
        cv2.putText(buffer_img, text, curpt, cv2.FONT_ITALIC, 0.3, (0, 255, 0), 1)
        cv2.rectangle(buffer_img, prept, curpt, (255, 0, 0), 1)
        cv2.imshow('buffer_img', buffer_img)
    # as left button has been down and mouse is moving, draw rectangle
    elif (event == cv2.EVENT_MOUSEMOVE) & (flags & cv2.EVENT_FLAG_LBUTTON):
        buffer_img = np.copy(origin_img)
        text = '({},{})'.format(x, y)
        # cv2.circle(buffer_img, prept, 2, (0, 0, 255), 2)
        cv2.putText(buffer_img, text, (x, y), cv2.FONT_ITALIC, 0.3, (0, 255, 0), 1)
        cv2.rectangle(buffer_img, prept, (x, y), (255, 0, 0), 1)
        cv2.imshow('buffer_img', buffer_img)









