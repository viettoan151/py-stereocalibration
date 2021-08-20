import numpy as np
import glob
import argparse
import os
from tqdm import tqdm
import random
import yaml
import matplotlib.pyplot as plt
import cv2

# actual position in EuRoC is left:cam1, right:cam0
CAMFOLDER = {'left': 'cam0/data',
             'right': 'cam1/data'}
IMGEXT = '.png'

CHESSBOARD_SIZE = (7, 6)  # (w, h) For EuRoC Cols:Rows is 6:7 internal corners

SQUARE_SIZE = 0.06  # size of one chessboard square [m]
USE_GOODIMG = True
SELECT_GOODIMG = False

GOODIMG_FILES = {'left': 'euroc/Selected_Good_stereo_left_image_file.txt',
                 'right': 'euroc/Selected_Good_stereo_right_image_file.txt'}

RANDSEED = 12345
# Number of images that used in calibration
FEATURE_NUM = 40


class StereoCalibration(object):
    def __init__(self, filepath):
        # termination criteria
        self.criteria_sub = (cv2.TERM_CRITERIA_EPS +
                             cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS +
                             cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.ckb_objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
        self.ckb_objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
        self.ckb_objp = self.ckb_objp * SQUARE_SIZE
        
        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.
        
        if os.path.splitext(filepath)[-1] == '.yaml':
            print('initial from calibrated file ' + filepath)
            self._load_stereo_model(filepath)
            
        else:
            self.cal_path = filepath
            self.collect_pair_images(self.cal_path)
            self.extract_feature_points()
            self.stereo_calibrate()
            
        self._generate_rectify_map()
    
    @staticmethod
    def _create_random_list(source_len, target_len):
        # random array of index elements
        if target_len > 10:
            random.seed(RANDSEED)
            idx_rand = random.sample(range(source_len), target_len)
        else:
            print('Number of feature = %d is too small' % target_len)
            print('All files will be used to calibrate stereo')
            idx_rand = range(source_len)
        return idx_rand
    
    def _load_stereo_model(self, file_path):
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        self.img_shape = tuple(data['dimension'])
        self.camera_model = data['camera_model']
        self.rectify_model = data['rectify_model']
        self.stereo_parameters = dict([('camera_model', self.camera_model),
                                       ('rectify_model', self.rectify_model)])

    
    def _read_good_images_list(self):
        self.images_left = []
        self.images_right = []
        # read good images files
        with open(GOODIMG_FILES['left'], 'r') as f:
            self.images_left = f.readlines()
        with open(GOODIMG_FILES['right'], 'r') as f:
            self.images_right = f.readlines()
            
        self.images_left = [fi.strip() for fi in self.images_left]
        self.images_right = [fi.strip() for fi in self.images_right]
        
        # sort images
        self.images_left.sort()
        self.images_right.sort()
        return self.images_left, self.images_right

    def collect_pair_images(self, cal_path):
        self.images_left = []
        self.images_right = []
        if not USE_GOODIMG:
            # these getting images is for EuRoC calibration folder
            print('Read list all images in folders')
            self.images_left = glob.glob(os.path.join(cal_path, CAMFOLDER['left']) + '/*' + IMGEXT)
            self.images_right = glob.glob(os.path.join(cal_path, CAMFOLDER['right']) + '/*' + IMGEXT)
        else:
            print('List all images from good images files')
            self._read_good_images_list()
            
            if SELECT_GOODIMG:
                # image will be selected later
                self.images_left = [img.strip() for img in self.images_left]
                self.images_right = [img.strip() for img in self.images_right]
            
            else:
                # randomly select images by array of index elements
                rand_idx = self._create_random_list(len(self.images_left), FEATURE_NUM)
                print('Randomly select good images {} indices:'.format(FEATURE_NUM))
                print(rand_idx)
                self.images_left = [self.images_left[idx].strip() for idx in rand_idx]
                self.images_right = [self.images_right[idx].strip() for idx in rand_idx]
        
        # sort images
        self.images_left.sort()
        self.images_right.sort()
    
    def extract_feature_points(self):
        # Check whether or not enough the number of images
        if (len(self.images_left) < 10) or (len(self.images_right) < 10):
            print("ERROR: don\'t have enough images. Left: %d, Right: %d" % (len(self.images_left), len(self.images_right)))
            return
        else:
            print("Found {} images of left camera".format(len(self.images_left)))
            print("Found {} images of right camera".format(len(self.images_right)))
        
        good_image_l = []
        good_image_r = []
        good_idx = []
        # Get a sample image shape
        self.img_shape = cv2.imread(self.images_left[0]).shape[:2]
        # shape is reversed for use in opencv function
        self.img_shape = tuple(reversed(self.img_shape))
        print('Image shape:', self.img_shape)
        
        # Process to corners detection and extract feature points
        total_img = min(len(self.images_left), len(self.images_right))
        pbar = tqdm(desc='Find chessboard:', total=total_img, ncols=100, unit='image')
        for idx in range(total_img):
            img_l = cv2.imread(self.images_left[idx])
            img_r = cv2.imread(self.images_right[idx])
            
            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
            
            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHESSBOARD_SIZE,
                                                         flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHESSBOARD_SIZE,
                                                         flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
            
            # If found, add object points, image points (after refining them)
            if ret_l and ret_r:
                # refine and add left image points
                corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11),
                                      (-1, -1), self.criteria_sub)
                
                # refine and add right image points
                corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11),
                                      (-1, -1), self.criteria_sub)
                
                # Draw and display the corners
                ret_l = cv2.drawChessboardCorners(img_l, CHESSBOARD_SIZE,
                                                  corners_l, ret_l)
                ret_r = cv2.drawChessboardCorners(img_r, CHESSBOARD_SIZE,
                                                  corners_r, ret_r)
                cv2.imshow('left', img_l)
                cv2.imshow('right', img_r)

                # Good images can be selected again by key 's'
                if SELECT_GOODIMG:
                    print('Press \'s\' to select this image pair')
                    pkey = cv2.waitKey(0)
                    if pkey & 0xFF == ord('s'):
                        good_image_l.append(self.images_left[idx])
                        good_image_r.append(self.images_right[idx])
                        
                        # add corner points and associated object points
                        self.imgpoints_l.append(corners_l)
                        self.imgpoints_r.append(corners_r)
                        self.objpoints.append(self.ckb_objp)
                        
                        print('Selected %d good images over %d total images' % (len(good_image_l), total_img))
                else:
                    good_image_l.append(self.images_left[idx])
                    good_image_r.append(self.images_right[idx])

                    # add corner points and associated object points
                    self.imgpoints_l.append(corners_l)
                    self.imgpoints_r.append(corners_r)
                    self.objpoints.append(self.ckb_objp)
                    
                    cv2.waitKey(1)
            # update process bar
            pbar.update()

        # end extracting process
        pbar.close()
        cv2.destroyAllWindows()
        
        print("Found {} pairs".format(len(self.imgpoints_r)))
        
        # write down selected good image list
        if SELECT_GOODIMG:
            with open('Selected_' + GOODIMG_FILES['left'], 'w') as f:
                for item in good_image_l:
                    f.write("%s\n" % item)
            with open('Selected_' + GOODIMG_FILES['right'], 'w') as f:
                for item in good_image_r:
                    f.write("%s\n" % item)
            print('Finished selecting good image file!')
            return
        
        # write down good image list
        if not USE_GOODIMG:
            with open(GOODIMG_FILES['left'], 'w') as f:
                for item in good_image_l:
                    f.write("%s\n" % item)
            with open(GOODIMG_FILES['right'], 'w') as f:
                for item in good_image_r:
                    f.write("%s\n" % item)
    
    def stereo_calibrate(self):
        # stereo calibration flags
        flags = int(0)
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        # flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5
        
        assert (len(self.objpoints) > 10), print('Don\'t have enough feature points')

        # random array of index elements
        idx_rand = self._create_random_list(len(self.objpoints), FEATURE_NUM)
        print(idx_rand)
        imgpoints_l = [self.imgpoints_l[i] for i in idx_rand]
        imgpoints_r = [self.imgpoints_r[i] for i in idx_rand]
        objpoints = [self.objpoints[i] for i in idx_rand]
        print('Reduce number of pairs to:', FEATURE_NUM)
        
        print('Pre calibrate cameras intrinsic ...')
        rt, M1, d1, r1, t1 = cv2.calibrateCamera(objpoints, imgpoints_l,
                                                 self.img_shape, None, None)
        rt, M2, d2, r2, t2 = cv2.calibrateCamera(objpoints, imgpoints_r,
                                                 self.img_shape, None, None)
        
        # self.M1 = cv2.initCameraMatrix2D(objpoints, imgpoints_l, self.img_shape)
        # self.M2 = cv2.initCameraMatrix2D(objpoints, imgpoints_r, self.img_shape)
        # self.d1=np.array([0.0]*5)
        # self.d2=np.array([0.0]*5)
        
        print('Stereo calibration ...')
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
                                                objpoints, imgpoints_l, imgpoints_r,
                                                M1, d1, M2, d2, self.img_shape,
                                                criteria=self.criteria_cal, flags=flags)

        # ret, M2, d2, M1, d1, R, T, E, F = cv2.stereoCalibrate(
        #                                         objpoints, imgpoints_r, imgpoints_l,
        #                                         M2, d2, M1, d1, self.img_shape,
        #                                         criteria=self.criteria_cal, flags=flags)
        
        print("Total error:", ret)
        print('Intrinsic_mtx_1', M1)
        print('dist_1', d1)
        print('Intrinsic_mtx_2', M2)
        print('dist_2', d2)
        print('R', R)
        print('T', T)
        print('E', E)
        print('F', F)
        
        # for i in range(len(self.r1)):
        #     print("--- pose[", i+1, "] ---")
        #     self.ext1, _ = cv2.Rodrigues(self.r1[i])
        #     self.ext2, _ = cv2.Rodrigues(self.r2[i])
        #     print('Ext1', self.ext1)
        #     print('Ext2', self.ext2)
        
        print('')
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(M1, d1, M2, d2, self.img_shape, R, T)
        # R2, R1, P2, P1, Q, validPixROI2, validPixROI1  = cv2.stereoRectify( M2, d2,M1, d1, self.img_shape, R, T)
        
        # baseline by fx
        # b_fx = T[0, 0] * M1[0, 0] * SQUARE_SIZE
        #
        b_fx = T[0, 0]*M1[0, 0] # equal to P2[0][3]
        
        self.camera_model = dict([('M1', M1.tolist()), ('M2', M2.tolist()),
                                ('dist1', d1.tolist()), ('dist2', d2.tolist()),
                                ('R', R.tolist()), ('T', T.tolist()),
                                ('E', E.tolist()), ('F', F.tolist())])
        
        self.rectify_model = dict([('R1', R1.tolist()), ('R2', R2.tolist()),
                                   ('P1', P1.tolist()), ('P2', P2.tolist()),
                                   ('Q', Q.tolist()), ('ROI1', list(validPixROI1)),
                                   ('ROI2', list(validPixROI2)), ('b_fx', b_fx.tolist())])
        
        self.stereo_parameters = dict([('dimension', list(self.img_shape)),
                                       ('camera_model', self.camera_model),
                                       ('rectify_model', self.rectify_model)])
        
        print('Baseline is:', b_fx)
        with open('euroc/Stereo_calibration.yaml', 'w') as sf:
            print("Save stereo rectified parameter to file...")
            # json.dump(stereo_par, sf, indent=2)
            yaml.dump(self.stereo_parameters, sf)
        
        
    
    def _generate_rectify_map(self):
        self.leftMapX, self.leftMapY = cv2.initUndistortRectifyMap(
                np.asarray(self.camera_model['M1']), np.asarray(self.camera_model['dist1']),
                np.asarray(self.rectify_model['R1']), np.asarray(self.rectify_model['P1']),
                self.img_shape, cv2.CV_32FC1)
        self.rightMapX, self.rightMapY = cv2.initUndistortRectifyMap(
                np.asarray(self.camera_model['M2']), np.asarray(self.camera_model['dist2']),
                np.asarray(self.rectify_model['R2']), np.asarray(self.rectify_model['P2']),
                self.img_shape, cv2.CV_32FC1)
        
    
    def calculate_depth_map(self, img_left, img_right):
        stereoMatcher = cv2.StereoBM_create()
        stereoMatcher.setMinDisparity(4)
        stereoMatcher.setNumDisparities(128)
        stereoMatcher.setBlockSize(21)
        stereoMatcher.setSpeckleRange(16)
        stereoMatcher.setSpeckleWindowSize(45)
        
        leftFrame = cv2.imread(img_left)
        rightFrame = cv2.imread(img_right)
        
        fixedLeft = cv2.remap(leftFrame, self.leftMapX, self.leftMapY, cv2.INTER_LINEAR)
        fixedRight = cv2.remap(rightFrame, self.rightMapX, self.rightMapY, cv2.INTER_LINEAR)
    
        grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
        grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
        disparity = stereoMatcher.compute(grayLeft, grayRight)
        # disparity = stereoMatcher.compute(grayRight, grayLeft)
        baselinefx = abs(self.rectify_model['b_fx'])
        depth = baselinefx/disparity
        
        return disparity, depth
    

    def calculate_depth_map_v2(self, img_left, img_right):
        leftFrame = cv2.imread(img_left)
        rightFrame = cv2.imread(img_right)
    
        fixedLeft = cv2.remap(leftFrame, self.leftMapX, self.leftMapY, cv2.INTER_LINEAR)
        fixedRight = cv2.remap(rightFrame, self.rightMapX, self.rightMapY, cv2.INTER_LINEAR)

        grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
        grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
        
        # disparity range is tuned for 'aloe' image pair
        window_size = 3
        min_disp = 4
        num_disp = 68 - min_disp
        stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                      numDisparities=num_disp,
                                      blockSize=16,
                                      P1=8 * 3 * window_size ** 2,
                                      P2=32 * 3 * window_size ** 2,
                                      disp12MaxDiff=1,
                                      uniquenessRatio=10,
                                      speckleWindowSize=100,
                                      speckleRange=32
                                      )

        print('computing disparity...')
        disparity = stereo.compute(grayLeft, grayRight).astype(np.float32) / 16.0
        # disparity = stereoMatcher.compute(grayRight, grayLeft)
        baselinefx = abs(self.rectify_model['b_fx'])
        depth = baselinefx / disparity
    
        return disparity, depth

    def rectify_pair_images(self, img_left, img_right):
        isVerticalStereo = abs(np.asarray(self.rectify_model['P2'])[1,3]) > abs(np.asarray(self.rectify_model['P2'])[0,3])
        leftFrame = cv2.imread(img_left)
        rightFrame = cv2.imread(img_right)
        img_width = leftFrame.shape[1]
        img_height = leftFrame.shape[0]
        if not isVerticalStereo:
            # sf = 600. / max(img_width, img_height)
            sf = 1.0
            w = round(img_width * sf)
            h = round(img_height * sf)
            # canvas.create(h, w * 2, cv2.CV_8UC3)
        else:
            # sf = 300. / max(img_width, img_height)
            sf = 1.0
            w = round(img_width * sf)
            h = round(img_height * sf)
            # canvas.create(h * 2, w, cv2.CV_8UC3)

        fixedLeft = cv2.remap(leftFrame, self.leftMapX, self.leftMapY, cv2.INTER_LINEAR)
        fixedRight = cv2.remap(rightFrame, self.rightMapX, self.rightMapY, cv2.INTER_LINEAR)
        
        fixedLeft = cv2.resize(fixedLeft, (w,h), interpolation=cv2.INTER_AREA)
        fixedRight = cv2.resize(fixedRight, (w,h), interpolation=cv2.INTER_AREA)
        
        # write text to images
        cv2.putText(fixedLeft, 'Left',
                    (10,50),                    # position
                    cv2.FONT_HERSHEY_SIMPLEX,   # font
                    1,                          #
                    (255,255,255),              # color
                    2)                          # lineType

        cv2.putText(fixedRight, 'Right',
                    (10,50),                    # position
                    cv2.FONT_HERSHEY_SIMPLEX,   # font
                    1,                          #
                    (255,255,255),              # color
                    2)                          # lineType
        # draw valid ROI
        lROI = np.asarray(self.rectify_model['ROI1']) * sf
        rROI = np.asarray(self.rectify_model['ROI2']) * sf
        cv2.rectangle(fixedLeft, lROI, (255,0,0), 1, 8)
        cv2.rectangle(fixedRight, rROI, (255,0,0), 1, 8)

        if not isVerticalStereo:
            final_img = np.concatenate([fixedLeft, fixedRight], axis=1)
            for j in range(0, final_img.shape[0], 16):
                cv2.line(final_img, (0, j), (final_img.shape[1],j), (0, 255, 0), 1, 8)
        else:
            final_img = np.concatenate([fixedLeft, fixedRight], axis=0)
            for j in range(0, final_img.shape[1], 16):
                cv2.line(final_img, (j, 0), (j,final_img.shape[0]), (0, 255, 0), 1, 8)
                
        return final_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str,
                        # default='EuroC/cam_checkerboard/mav0',
                        default='euroc/Stereo_calibration.yaml',
                        help='String Filepath')
    
    args = parser.parse_args()
    stereo_obj = StereoCalibration(args.filepath)
    # stereo_obj = StereoCalibration(".")
    left_list, right_list = stereo_obj._read_good_images_list()
    for idx in range(len(left_list)):
        rect_img = stereo_obj.rectify_pair_images(left_list[idx], right_list[idx])
        disp_img, depth_img = stereo_obj.calculate_depth_map_v2(left_list[idx], right_list[idx])
        plt.clf()
        plt.imshow(depth_img)
        plt.draw()
        plt.pause(0.001)
        # plt.imshow(disp_img)
        cv2.imshow("rectified", rect_img)
        key = cv2.waitKey(0)
        if key & 0xFF == ord('q'):
            break
