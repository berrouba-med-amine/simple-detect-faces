"""
This simple example show how to detect faces using opencv in python

Author: Berrouba.A
Last edited: 24 Feb 2018
"""

# import system module
import sys

# import argparse module
import argparse

# import Opencv module
import cv2

# detect face
def detectFaces(img):

    # load face cascade classifier
    face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')
    if face_cascade.empty():
        print("Error Loading cascade classifier", "Unable to load the face	cascade classifier xml file")
        sys.exit()

    # clone image in frame
    frame = img[:,:,:]

    # convert frame to GRAY format
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect rect faces
    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

    # for all faces
    for (x, y, w, h) in face_rects:
        # draw green rect on face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # print number of detected faces
    print('{} face(s) detected.'.format(len(face_rects)))

    # return frame with detected feces
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This is a simple face detection using python and Opencv3.')
    parser.add_argument('-i', '--input', help='Input file name image', required=True)
    parser.add_argument('-o', '--output', help='Output file name image', required=False)
    args = parser.parse_args()

    # read input image
    img = cv2.imread(args.input)

    # detect faces
    frame = detectFaces(img)

    # show detected faces
    cv2.imshow('Output image', frame)

    # save output image if demanded
    if args.output:
        try:
            cv2.imwrite(args.output, frame)
            print('image with detected faces saved in {}'.format(args.output))
        except cv2.error as e:
            print(e)
    cv2.waitKey(0)

