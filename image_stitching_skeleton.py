#
# CSCI3290 Computational Imaging and Vision *
# --- Declaration --- *
# I declare that the assignment here submitted is original except for source
# material explicitly acknowledged. I also acknowledge that I am aware of
# University policy and regulations on honesty in academic work, and of the
# disciplinary guidelines and procedures applicable to breaches of such policy
# and regulations, as contained in the website
# http://www.cuhk.edu.hk/policy/academichonesty/ *
# Assignment 2
# Name : Wong Kai Long
# Student ID : 1155096748
# Email Addr : 1155096748@link.cuhk.edu.hk
#

import cv2
import numpy as np
import argparse


def extract_and_match_feature(img_1, img_2, ratio_test=0.7):
    """
    1/  extract SIFT feature from image 1 and image 2,
    2/  use a bruteforce search to find pairs of matched features:
        for each feature point in img_1, find its best matched feature point in img_2
    3/ apply ratio test to select the set of robust matched points

    :param img_1: input image 1
    :param img_2: input image 2
    :param ratio_test: ratio for the robustness test
    :return list_pairs_matched_keypoints: a list of pairs of matched points: [[[p1x,p1y],[p2x,p2y]]]
    """
    list_pairs_matched_keypoints = []
    gray_1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
    sift_1 = cv2.xfeatures2d.SIFT_create()
    kp_1, des_1 = sift_1.detectAndCompute(gray_1,None)
    #img_1=cv2.drawKeypoints(gray_1,kp_1,img_1)


    gray_2 = cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY)
    sift_2 = cv2.xfeatures2d.SIFT_create()
    kp_2, des_2 = sift_2.detectAndCompute(gray_2,None)
    #img_2=cv2.drawKeypoints(gray_2,kp_2,img_2)


    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_1, des_2, k=2)

    list_kp1 = []
    list_kp2 = []
    for m,n in matches:
        if m.distance < ratio_test*n.distance:

            img1_idx = m.queryIdx
            img2_idx = m.trainIdx
            (x1, y1) = kp_1[img1_idx].pt
            (x2, y2) = kp_2[img2_idx].pt
            list_kp1.append([x1, y1])
            list_kp2.append([x2, y2])

    for i in range(len(list_kp1)):
        list_pairs_matched_keypoints.append([list_kp1[i],list_kp2[i]])

    # to be completed ....


    return list_pairs_matched_keypoints


def find_homography_ransac(list_pairs_matched_keypoints,
                           threshold_ratio_inliers=0.85,
                           threshold_reprojection_error=3,
                           max_num_trial=1000):
    """
    Apply RANSAC algorithm to find a homography transformation matrix that align 2 sets of feature points,
    transform the second set of feature point to the first (e.g. warp image 2 to image 1)

    :param list_pairs_matched_keypoints: a list of pairs of matched points: [[[p1x,p1y],[p2x,p2y]],...]
    :param threshold_ratio_inliers: threshold on the ratio of inliers over the total number of samples,
                                    accept the estimated homography if ratio is higher than the threshold
    :param threshold_reprojection_error: threshold of reprojection error (measured as euclidean distance, in pixels)
                                            to determine whether a sample is inlier or outlier
    :param max_num_trial: the maximum number of trials to take sample and do testing to find the best homography matrix
    :return best_H: the best found homography matrix
    """

    best_H = None
    src = []
    dst = []
    #src = np.float32([ list_pairs_matched_keypoints[0] ]).reshape(-1,1,2)
    #dst = np.float32([ list_pairs_matched_keypoints[1] ]).reshape(-1,1,2)
    #print(len(list_pairs_matched_keypoints))
    for i in range (len(list_pairs_matched_keypoints)):
        src.append(list_pairs_matched_keypoints[i][0])
        dst.append(list_pairs_matched_keypoints[i][1])
    src = np.asarray(src)
    dst = np.asarray(dst)
    # to be completed ...
    best_H, mask = cv2.findHomography(src, dst, cv2.RANSAC, threshold_reprojection_error, None, max_num_trial, threshold_ratio_inliers)

    return best_H


def warp_blend_image(img_1, H, img_2):
    """
    1/  warp image img_2 using the homography H to align it with image img_1
        (using inverse warping and bilinear resampling)
    2/  stitch image img_2 to image img_1 and apply average blending to blend the 2 images into a single panorama image

    :param img_1:  the original first image
    :param H: estimated homography
    :param img_2:the original second image
    :return img_panorama: resulting panorama image
    """
    img_panorama = None
    # to be completed ...
    H = np.linalg.inv(H)
    row = img_2.shape[1]+img_2.shape[1]
    col = img_2.shape[0]+img_2.shape[0]
    wrap = cv2.warpPerspective(img_2, H, (row , col))

    wrap[0:img_2.shape[0], 0:img_2.shape[1]] = img_1

    rows, cols = np.where(wrap[:,:,0] !=0)

    img_panorama = wrap[0:max(rows) +1,0:max(cols) +1,:]
    return img_panorama


def stitch_images(img_1, img_2):
    """
    :param img_1: input image 1 is the reference image. We will not warp this image
    :param img_2: We warp this image to align and stich it to the image 1
    :return img_panorama: the resulting stiched image
    """
    print('==================================================================================')
    print('===== stitch two images to generate one panorama image =====')
    print('==================================================================================')

    # ===== extract and match features from image 1 and image 2
    list_pairs_matched_keypoints = extract_and_match_feature(img_1=img_1, img_2=img_2, ratio_test=0.7)

    # ===== use RANSAC algorithm to find homography to warp image 2 to align it to image 1
    H = find_homography_ransac(list_pairs_matched_keypoints, threshold_ratio_inliers=0.85,threshold_reprojection_error=3, max_num_trial=1000)

    # ===== warp image 2, blend it with image 1 using average blending to produce the resulting panorama image
    img_panorama = warp_blend_image(img_1=img_1, H=H, img_2=img_2)


    return img_panorama


if __name__ == "__main__":
    print('==================================================================================')
    print('CSCI3290, Spring 2020, Assignment 2: image stitching')
    print('==================================================================================')

    parser = argparse.ArgumentParser(description='Image Stitching')
    parser.add_argument('--im1', type=str, default='test_images/MelakwaLake1.png',
                        help='path of the first input image')
    parser.add_argument('--im2', type=str, default='test_images/MelakwaLake2.png',
                        help='path of the second input image')
    parser.add_argument('--output', type=str, default='MelakwaLake.png',
                        help='the path of the output image')
    args = parser.parse_args()

    # ===== read 2 input images
    img_1 = cv2.imread(args.im1)
    img_2 = cv2.imread(args.im2)

    # ===== create a panorama image
    img_panorama = stitch_images(img_1=img_1, img_2=img_2)

    # ===== save panorama image
    cv2.imwrite(filename=args.output, img=img_panorama.clip(0.0, 255.0).astype(np.uint8))
