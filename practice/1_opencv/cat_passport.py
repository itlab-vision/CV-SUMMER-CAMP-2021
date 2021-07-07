# python cat_passport.py -m haarcascade_frontalcatface.xml -i cat.jpg

import argparse
import sys
import cv2
from random import randrange


def make_cat_passport_image(input_image_path, haar_model_path):

    # Read image
    image = cv2.imread(input_image_path)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize image intensity
    gray = cv2.equalizeHist(gray)

    # Resize image
    resized = cv2.resize(gray, (640, 480), interpolation = cv2.INTER_AREA)

    # Detect cat faces using Haar Cascade
    detector = cv2.CascadeClassifier(haar_model_path)
    rects = detector.detectMultiScale(image,
                                      scaleFactor=1.1,
                                      minNeighbors=5,
                                      minSize=(75, 75))


    # Draw bounding box
    for (i, (x, y, w, h)) in enumerate(rects):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image,
                    "Cat #{}".format(i + 1),
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2)

    # Display result image

    """
    cv2.imshow("window_name", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    # Crop image
    x, y, w, h = rects[0]
    image = image[y:y+h, x:x+w]

    # Save result image to file
    cv2.imwrite('out.jpg', image)

    # Additional task
    image_final = cv2.imread('pet_passport.png')
    image_final[49:(49+138), 49:(49+138)] = image

    # Name
    cv2.putText(image_final,
                'Murka',
                (91, 219),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255,0,0),
                1)

    # Species
    cv2.putText(image_final,
                'Cat',
                (91, 232),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 0, 0),
                1)

    # Breed
    cv2.putText(image_final,
                'Siamese Cat',
                (91, 246),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 0, 0),
                1)

    # Sex
    cv2.putText(image_final,
                'Female',
                (91, 260),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 0, 0),
                1)

    # Date of Birth
    cv2.putText(image_final,
                '01.01.2020',
                (116, 273),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 0, 0),
                1)

    # Tattoo number
    tattoo_number = randrange(100000, 999999)
    cv2.putText(image_final,
                str(tattoo_number),
                (274, 212),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 0, 0),
                1)

    # Date of Tattoing
    cv2.putText(image_final,
                '11.01.2020',
                (274, 253),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 0, 0),
                1)

    cv2.imwrite('out_final.jpg', image_final)

    return


def build_argparser():
    parser = argparse.ArgumentParser(
        description='Speech denoising demo', add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument('-m', '--model', type=str, required=True,
                      help='Required. Path to .XML file with pre-trained model.')
    args.add_argument('-i', '--input', type=str, required=True,
                      help='Required. Path to input image')
    return parser


def main():

    args = build_argparser().parse_args()
    make_cat_passport_image(args.input, args.model)

    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
