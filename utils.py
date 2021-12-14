import cv2 as cv
import glob
import os
import numpy as np

class Q1():
    def __init__(self, path:str):
        self.images = []
        self.count = []
        for path_image in glob.glob(os.path.join(path, "*.jpg")):
            image = cv.imread(path_image)
            h,w = image.shape[:2]
            # print(h, w)
            image_resize = cv.resize(image, (w//2, h//2), interpolation=cv.INTER_CUBIC)
            # print(image_resize.shape[:2])
            self.images.append([image_resize, path_image])
        pass

    def find_contour(self, show = True):
        for image, path_image in self.images:
            gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            h, w = gray_img.shape
            blur_img = cv.GaussianBlur(gray_img, (7,7), 1)
            binary_img = cv.adaptiveThreshold(blur_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv.THRESH_BINARY, 199, 17)
            edges = cv.Canny(binary_img,50,255)
            contours, hierarchy = cv.findContours(edges, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
            contour_img = image.copy()
            count_rings = []
            for contour in contours:
                area = cv.contourArea(contour)
                if area > ((w//15) * (h//15)):
                    contour_img = cv.drawContours(contour_img, contour, -1, (0,255,0), 2)
                    count_rings.append(contour)

            # gaussian_img = cv.GaussianBlur(image, 3, 5)
            # print(len(count_rings)//4)
            self.count.append(len(count_rings)//4)
            if show:
                cv.namedWindow(path_image, cv.WINDOW_GUI_EXPANDED)
                cv.imshow(path_image, contour_img)
                cv.waitKey(0)
        pass

def readImages(pathFolder:str, extension = "*.bmp"):
    images = []
    for path_image in glob.glob(os.path.join(pathFolder, extension)):
        image = cv.imread(path_image)
        if image is not None:
            images.append(image)
    return images

def concat_image(src, dst):
    # merge = np.concatenate((src, dst), axis=1)
    merge = cv.hconcat([src, dst])
    return merge

def calibration(images:list, width_board = 11, height_board = 8):
    
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points
    objp = np.zeros((height_board*width_board, 3), np.float32)
    objp[:,:2] = np.mgrid[0:width_board, 0:height_board].T.reshape(-1, 2)

    # Array to store object points and image points from all the image.
    objpoints = [] #3d point in real world space
    imgpoints = [] #2d points in image plane

    for index, image in enumerate(images):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        ret, corner = cv.findChessboardCorners(gray, (width_board, height_board), None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corner)
    return objpoints, imgpoints


def init_feature(name):
    """
    The features include orb, akaza, brisk
    """
    FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
    FLANN_INDEX_LSH    = 6

    # if name == 'sift':
    #     detector = cv.xfeatures2d.SIFT_create()
    #     norm = cv.NORM_L2
    # elif name == 'surf':
    #     detector = cv.xfeatures2d.SURF_create(800)
    #     norm = cv.NORM_L2
    if name == 'orb':
        detector = cv.ORB_create(400)
        norm = cv.NORM_HAMMING
    elif name == 'akaze':
        detector = cv.AKAZE_create()
        norm = cv.NORM_HAMMING
    elif name == 'brisk':
        detector = cv.BRISK_create()
        norm = cv.NORM_HAMMING
    else:
        return None, None
    if 'flann' in name:
        if norm == cv.NORM_L2:
            flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        else:
            flann_params= dict(algorithm = FLANN_INDEX_LSH,
                               table_number = 6, # 12
                               key_size = 12,     # 20
                               multi_probe_level = 1) #2
        matcher = cv.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    else:
        matcher = cv.BFMatcher(norm)
    return detector, matcher

def filter_matches(kp1, kp2, matches, ratio = 0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, list(kp_pairs)

def explore_match(image_1, image_2, keypoint_pair, status = None, H = None):
    h1, w1 = image_1.shape[:2]
    h2, w2 = image_2.shape[:2]
    image_match = np.zeros((max(h1, h2), w1+w2, 3), np.uint8)
    image_match[:h1, :w1] = image_1
    image_match[:h2, w1: w1+w2] = image_2

    if H is not None:
        corners = np.float32([[0,0], [w1, 0], [w1,h1], [0,h1] ])
        corners = np.int32(cv.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0))
        cv.polylines(image_match, [corners], True, (255,255,255), 6)

    if status is None:
        status = np.ones(len(keypoint_pair), np.bool_)
    point_1, point_2 = [], []

    for kpp in keypoint_pair:
        point_1.append(np.int32(kpp[0].pt))
        point_2.append(np.int32(np.array(kpp[1].pt) + [w1, 0]))

    green = (0,255,0)
    red = (0,0,255)
    keypoint_color = (125, 108, 96)

    for (x1, y1), (x2, y2), inlier in zip(point_1, point_2, status):
        if inlier:
            color = green
            cv.circle(image_match, (x1,y1), 2, color, -1)
            cv.circle(image_match, (x2,y2), 2, color, -1)
        else:
            color = red
            r = 2
            thickness = 3
            cv.line(image_match, (x1-r, y1-r), (x1+r, y1+r), color, thickness)
            cv.line(image_match, (x1-r, y1+r), (x1+r, y1-r), color, thickness)
            cv.line(image_match, (x2-r, y2-r), (x2+r, y2+r), color, thickness)
            cv.line(image_match, (x2-r, y2+r), (x2+r, y2-r), color, thickness)
    image_match0 = image_match.copy()
    for (x1, y1), (x2, y2), inlier in zip(point_1, point_2, status):
        if inlier:
            color = green
            cv.line(image_match, (x1,y1), (x2,y2), green)
    return image_match

def draw2(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    #img = cv2.drawContours(img, [imgpts[1:]],-1,(0,255,0),-3)
    img = cv.line(img, tuple(imgpts[1]), tuple(imgpts[2]), (0, 0, 255), 3)
    img = cv.line(img, tuple(imgpts[1]), tuple(imgpts[3]),  (0, 0, 255), 3)
    img = cv.line(img, tuple(imgpts[2]), tuple(imgpts[3]), (0, 0, 255), 3)
    # draw pillars in blue color
    # for i,j in zip(range(3),range(1,3)):
    img = cv.line(img, tuple(imgpts[0]), tuple(imgpts[1]),(0, 0,255),3)
    img = cv.line(img, tuple(imgpts[0]), tuple(imgpts[2]), (0, 0, 255), 3)
    img = cv.line(img, tuple(imgpts[0]), tuple(imgpts[3]), (0, 0, 255), 3)
    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[:1]],-1,(0,0,255),3)
    img = cv.resize(img, (650, 650))  # Resize image
    return img

def draw_char(img, char_list:list):
    draw_image = img.copy()
    for line in char_list:
        line = line.reshape(2,2)
        draw_image = cv.line(draw_image, tuple(line[0]), tuple(line[1]), (0,255,0), 10, cv.LINE_AA)
    return draw_image    

def disparity(imgL, imgR):
    matcher = cv.StereoBM_create(256,25)
    disparity_f = matcher.compute(imgL, imgR)
    return disparity_f

def process_ouput(disparity):
    cv8uc = cv.normalize(disparity, None, alpha=0,
    beta=255, norm_type=cv.NORM_MINMAX,
    dtype=cv.CV_8UC1)
    return cv8uc

def map_disparity(imgL, imgR, disparity, win):
    hL, wL = imgL.shape[:2]
    # hR, wR = imgR.shape[:2]
    print("Shape image: {}, {}".format(wL, hL))
    merge_image = concat_image(imgL, imgR)
    # imag0 = merge_image.copy()
    cur_img = merge_image.copy()
    cv.namedWindow(win, cv.WINDOW_GUI_EXPANDED)
    cv.imshow(win, cur_img)
    def onmouse(event, x, y, flags, param):
        # print("initial")
        nonlocal cur_img
        if flags & cv.EVENT_FLAG_LBUTTON:
            cur_img = merge_image.copy()
            cur_x, cur_y = x, y
            if cur_x < 0:
                cur_x = 0
            elif cur_x >= wL:
                cur_x = wL-1
            if cur_y < 0:
                cur_y = 0
            elif cur_y >= hL:
                cur_y = hL-1
            delta_pos = disparity[cur_y,cur_x]
            print("disparity value at ({},{}): {}".format(cur_x, cur_y, delta_pos))
            if delta_pos > 0:
                x_right = cur_x -delta_pos + wL
                cur_img = cv.circle(cur_img, (x_right,cur_y), radius=20, color=(0, 255, 0), thickness=-1)
            
        cv.imshow(win, cur_img)
    cv.setMouseCallback(win, onmouse) 

if __name__ == "__main__":
    q0 = Q1(r"./Q1_Image")
    q0.find_contour()


