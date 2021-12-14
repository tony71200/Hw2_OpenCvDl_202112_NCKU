import sys
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import QLabel, QWidget, QLineEdit
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QPushButton, QGroupBox
from PyQt5.QtCore import Qt, QMetaObject

import cv2 as cv
import glob
import os
import numpy as np
import utils
__appname__ = "2021 Opencvdl Hw1 "

class windowUI(object):
    """
    Set up UI
    please don't edit
    """
    def setupUI(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(640,480)
        MainWindow.setWindowTitle(__appname__)

        # 1. Find Contour Group
        Find_contour_group = QGroupBox("1. Find Contour")
        group_V0_vBoxLayout = QVBoxLayout(Find_contour_group)

        self.button0_1 = QPushButton("1.1 Draw Contour")
        self.button0_2 = QPushButton("1.2 Count Rings")
        self.label0_2 = QLabel("There are _ rings in img1.jpg\nThere are _ rings in img2.jpg\n")

        group_V0_vBoxLayout.addWidget(self.button0_1)
        group_V0_vBoxLayout.addWidget(self.button0_2)
        group_V0_vBoxLayout.addWidget(self.label0_2)
        group_V0_vBoxLayout.addStretch(1)


        # 2. Calibration Group
        Calibration_Group = QGroupBox("2. Calibration")
        group_V1_vBoxLayout = QVBoxLayout(Calibration_Group)

        self.button1_1 = QPushButton("2.1 Find Corners")
        self.button1_2 = QPushButton("2.2 Find Intrinsic")

        Find_Extrinsic_Group = QGroupBox("2.3 Find Extrinsic")
        group_V1_3_vBoxLayout = QVBoxLayout(Find_Extrinsic_Group)

        select_layout, self.edit_1_3 = self.edit_Text("Select image: ")
        group_V1_3_vBoxLayout.addLayout(select_layout)
        self.button1_3 = QPushButton("2.3 Find Extrinsic")
        group_V1_3_vBoxLayout.addWidget(self.button1_3)
        
        self.button1_4 = QPushButton("2.4 Find Distortion")
        self.button1_5 = QPushButton("2.5 Show Result")

        group_V1_vBoxLayout.addWidget(self.button1_1)
        group_V1_vBoxLayout.addWidget(self.button1_2)
        group_V1_vBoxLayout.addWidget(Find_Extrinsic_Group)
        group_V1_vBoxLayout.addWidget(self.button1_4)
        group_V1_vBoxLayout.addWidget(self.button1_5)

        # 3. Augmented Reality Group
        Augmented_Reality_Group = QGroupBox("3. Augmented Reality")
        group_V2_vBoxLayout = QVBoxLayout(Augmented_Reality_Group)

        self.edit_2 = QLineEdit("OPENCV")
        self.button2_1 = QPushButton("3.1 Show Words on Board")
        self.button2_2 = QPushButton("3.2 Show Words Vertically")
        group_V2_vBoxLayout.addWidget(self.edit_2)
        group_V2_vBoxLayout.addWidget(self.button2_1)
        group_V2_vBoxLayout.addWidget(self.button2_2)
        
        
        # 4. Stereo Disparity Map Group
        Stereo_Disparity_Group = QGroupBox("4. Stereo Disparity Map")
        group_V3_vBoxLayout = QVBoxLayout(Stereo_Disparity_Group)

        # self.button3_1 = QPushButton("3.1 Stereo Disparity Map")
        self.button3_2 = QPushButton("3.1 Checking the Disparity Value")

        # group_V3_vBoxLayout.addWidget(self.button3_1)
        group_V3_vBoxLayout.addWidget(self.button3_2)
 

        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        vLayout = QHBoxLayout()
        vLayout.addWidget(Find_contour_group)
        vLayout.addWidget(Calibration_Group)
        vLayout.addWidget(Augmented_Reality_Group)
        vLayout.addWidget(Stereo_Disparity_Group)
        
        self.centralwidget.setLayout(vLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        QMetaObject.connectSlotsByName(MainWindow)

    @staticmethod
    def edit_Text(title:str, unit = "", showUnit= False):
        hLayout = QHBoxLayout()

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        title_label.setFixedWidth(60)
        unit_label = QLabel(unit)
        unit_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        unit_label.setFixedWidth(30)
        editText = QLineEdit("1")
        editText.setFixedWidth(50)
        editText.setAlignment(Qt.AlignRight)
        editText.setValidator(QIntValidator())

        hLayout.addWidget(title_label, alignment=Qt.AlignLeft)
        hLayout.addWidget(editText)
        if showUnit:
            hLayout.addWidget(unit_label)
        return hLayout, editText

class MainWindow(QMainWindow, windowUI):

    def __init__(self, parent = None):
        super(MainWindow, self).__init__(parent=parent)
        self.setupUI(self)
        self.initialValue()
        self.buildUi()

    def buildUi(self):
        self.button0_1.clicked.connect(self.find_contour)
        self.button0_2.clicked.connect(self.count_rings)
        self.button1_1.clicked.connect(self.find_corners)
        self.button1_2.clicked.connect(self.find_intrinsic)
        self.button1_3.clicked.connect(self.find_extrinsic)
        self.button1_4.clicked.connect(self.find_distortion)
        self.button1_5.clicked.connect(self.show_result)

        self.button2_1.clicked.connect(self.word_on_board)
        self.button2_2.clicked.connect(self.word_Vertical)

        # self.button3_1.clicked.connect(self.stereoDisparity)
        self.button3_2.clicked.connect(self.checkDisparity)    
    
    def initialValue(self):
        self.Q0 = utils.Q1(r"./Q1_Image")
        self.images_Q1 = []
        self.images_Q2 = []
        self.images_Q4 = []
        self.q0 = False
        self.q1_1 = False
        self.q1_2 = False
        self.q2_calib = False

        self.char_in_board = [ # coordinate for 6 charter in board (x, y) ==> (w, h)
            [7,5,0], # slot 1
            [4,5,0], # slot 2
            [1,5,0], # slot 3
            [7,2,0], # slot 4
            [4,2,0], # slot 5
            [1,2,0]  # slot 6
        ]
        self.q3_1 = False

    def find_contour(self):
        self.Q0.find_contour()
        self.q0 = True
        pass

    def count_rings(self):
        if not self.q0:
            self.Q0.find_contour(False)

        count = self.Q0.count
        if len(count) > 0:
            string = "There are {} rings in img1.jpg\nThere are {} rings in img2.jpg\n".format(*count)
            self.label0_2.setText(string)
        pass

    def find_corners(self):
        width = 11
        height = 8
        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare object points
        objp = np.zeros((height*width, 3), np.float32)
        objp[:,:2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

        # Array to store object points and image points from all the image.
        self.objpoints = [] #3d point in real world space
        self.imgpoints = [] #2d points in image plane
    
        self.images_Q1 = utils.readImages("Q2_Image")
        self.setEnabled(False)
        for image in self.images_Q1:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (width,height), None)

            # If found, add object points, image points
            if ret:
                self.objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1, -1), criteria)
                self.imgpoints.append(corners)

                # Draw and Display the corners
                find_corner_image = cv.drawChessboardCorners(image.copy(), (width,height), corners2, ret)
                QApplication.processEvents()

                cv.namedWindow("Find corners", cv.WINDOW_GUI_EXPANDED)
                cv.imshow("Find corners", find_corner_image)
                cv.waitKey(500)
        cv.destroyAllWindows()
        self.setEnabled(True)
        self.q1_1 = True


    def find_intrinsic(self):
        if not self.q1_1:
            self.find_corners()
        h, w = self.images_Q1[0].shape[:2]
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(self.objpoints, self.imgpoints, (w,h), None, None)
        QApplication.processEvents()
        print("Intrinsic matrix: \n", mtx)
        self.dist = dist
        self.rotate_v = rvecs
        self.translate_v = tvecs
        self.mtx = mtx
        self.q1_2 = True

    def find_extrinsic(self):
        if not self.q1_2:
            self.find_intrinsic()
        number_image = int(self.edit_1_3.text())
        if not ((number_image -1) < 0 or (number_image -1)>= len(self.images_Q1)):
            rvec = self.rotate_v[number_image - 1]
            tvec = self.translate_v[number_image-1]
            tvec = tvec.reshape(3,1)
            if rvec is not None and tvec is not None:
                Rotation_matrix = cv.Rodrigues(rvec)[0]
                Extrinsic_matrix = np.hstack([Rotation_matrix, tvec])
                print("Extrinsix: \n", Extrinsic_matrix)
        else:
            print("Input error: Please input from 1-15")
        
    def find_distortion(self):
        if not self.q1_2:
            self.find_intrinsic()
        print("Distortion: \n", self.dist[-1])

    def show_result(self):
        if not self.q1_2:
            self.find_intrinsic()
        self.setEnabled(False)
        for image in self.images_Q1:
            h, w = image.shape[:2]
            newcameramatrix, roi = cv.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w,h))
            dst = cv.undistort(image, self.mtx, self.dist, None, newcameramatrix)
            x, y, w, h = roi

            dst = dst[y:y+h, x:x+w]

            print(dst.shape)
            dst = cv.resize(dst, (image.shape[1], image.shape[0]))
            new = utils.concat_image(image, dst)
            QApplication.processEvents()
            cv.namedWindow("Show Result", cv.WINDOW_GUI_EXPANDED)
            cv.imshow("Show Result", new)
            cv.waitKey(500)
        self.setEnabled(True)
        cv.destroyAllWindows()
        pass

    def word_on_board(self):
        if len(self.images_Q2) == 0:
            self.images_Q2 = utils.readImages("Q3_Image")
        fs = cv.FileStorage("Q3_Image/Q3_Lib/alphabet_lib_onboard.txt", cv.FILE_STORAGE_READ)
        string = self.edit_2.text()[:6]
        if not string.isupper():
            string = string.upper()
        if not self.q2_calib:
            print("Calibrating>>>")
            self.q2_objps, self.q2_imageps = utils.calibration(self.images_Q2)
            QApplication.processEvents()
            self.q2_calib = True
        self.setEnabled(False)
        for index, image in enumerate(self.images_Q2):
            h, w = image.shape[:2]
            draw_image = image.copy()
            ret, intrinsic_mtx, dist, rvecs, tvecs = cv.calibrateCamera(self.q2_objps, self.q2_imageps, (w,h), None, None)
            QApplication.processEvents()
            if ret:
                rvec = np.array(rvecs[index])
                tvec = np.array(tvecs[index]).reshape(3,1)
                for i_char, character in enumerate(string):
                    ch = np.float32(fs.getNode(character).mat())
                    line_list = []
                    for eachline in ch:
                        ach = np.float32([self.char_in_board[i_char], self.char_in_board[i_char]])
                        eachline = np.add(eachline, ach)
                        image_points, jac = cv.projectPoints(eachline, rvec, tvec, intrinsic_mtx, dist)
                        line_list.append(image_points)
                    draw_image = utils.draw_char(draw_image, line_list)
                QApplication.processEvents()
                cv.namedWindow("WORD ON BOARD", cv.WINDOW_GUI_EXPANDED)
                cv.imshow("WORD ON BOARD", draw_image)
                cv.waitKey(800)
        self.setEnabled(True)   
        pass


    def word_Vertical(self):
        if len(self.images_Q2) == 0:
            self.images_Q2 = utils.readImages("Q3_Image")
        fs = cv.FileStorage("Q3_Image/Q3_Lib/alphabet_lib_vertical.txt", cv.FILE_STORAGE_READ)
        string = self.edit_2.text()[:6]
        if not string.isupper():
            string = string.upper()
        if not self.q2_calib:
            print("Calibrating>>>")
            self.q2_objps, self.q2_imageps = utils.calibration(self.images_Q2)
            QApplication.processEvents()
            self.q2_calib = True
        self.setEnabled(False)
        for index, image in enumerate(self.images_Q2):
            h, w = image.shape[:2]
            draw_image = image.copy()
            ret, intrinsic_mtx, dist, rvecs, tvecs = cv.calibrateCamera(self.q2_objps, self.q2_imageps, (w,h), None, None)
            QApplication.processEvents()
            if ret:
                rvec = np.array(rvecs[index])
                tvec = np.array(tvecs[index]).reshape(3,1)
                for i_char, character in enumerate(string):
                    ch = np.float32(fs.getNode(character).mat())
                    line_list = []
                    for eachline in ch:
                        ach = np.float32([self.char_in_board[i_char], self.char_in_board[i_char]])
                        eachline = np.add(eachline, ach)
                        image_points, jac = cv.projectPoints(eachline, rvec, tvec, intrinsic_mtx, dist)
                        line_list.append(image_points)
                    draw_image = utils.draw_char(draw_image, line_list)
                QApplication.processEvents()
                cv.namedWindow("WORD VERTICAL", cv.WINDOW_GUI_EXPANDED)
                cv.imshow("WORD VERTICAL", draw_image)
                cv.waitKey(800)   
        self.setEnabled(True)
        pass

    def stereoDisparity(self):
        self.setEnabled(False)
        self.imL = cv.imread("Q4_Image/imL.png")
        self.imR = cv.imread("Q4_Image/imR.png")

        grayL = cv.cvtColor(self.imL, cv.COLOR_BGR2GRAY)
        grayR = cv.cvtColor(self.imR, cv.COLOR_BGR2GRAY)

        disparity_f = utils.disparity(grayL, grayR)
        print(disparity_f.shape)
        self.u8 =utils.process_ouput(disparity_f) 
        self.setEnabled(True)
        #show disparity
        cv.namedWindow("Disparity", cv.WINDOW_GUI_EXPANDED)
        # cv.imshow("Disparity", (disparity - min_disp)/ num_disp)
        cv.imshow("Disparity", self.u8)
        cv.waitKey(1000)
        self.q3_1 = True
        pass

    def checkDisparity(self):
        if not self.q3_1:
            self.stereoDisparity()
        # cv.namedWindow("Checking Disparity", cv.WINDOW_GUI_EXPANDED)
        utils.map_disparity(self.imL, self.imR, self.u8, "Checking Disparity")
        cv.waitKey(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setGeometry(500, 150, 500, 300)
    window.show()
    sys,exit(app.exec_())

