import argparse
import cv2
import sys
import numpy as np

def make_cat_passport_image(input_image_path, haar_model_path):

    # Read image
    image = cv2.imread(input_image_path)
    passport = cv2.imread("pet_passport.png")

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize image intensity
    gray = cv2.equalizeHist(gray)
    # gray = cv2.normalizeHist(gray)

    # Resize image
    resized = cv2.resize(gray, (640, 480), interpolation=cv2.INTER_AREA)

    # Detect cat faces using Haar Cascade
    detector = cv2.CascadeClassifier(haar_model_path)
    rects = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(75, 75))

    # Draw bounding box
    for (i, (x, y, w, h)) in enumerate(rects):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, "Cat #{}".format(i + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # Display result image
    cv2.imshow("Full result", image)

    # Crop image
    x, y, w, h = rects[0]
    image = image[y:y+h, x:x+w]

    # Save result image to file
    cv2.imwrite('out.jpg', image)
    cv2.imshow("Created image", image)

    resized_face = cv2.resize(image, (170, 139), interpolation=cv2.INTER_AREA)
    passport[47:47 + 139, 30:30 + 170] = resized_face
    data = ['Moosya', 'Cat', '2', 'F', "12.11.2018", "White&black"]
    for ind, i in enumerate(range(217, 288, 14)): #263
        cv2.putText(passport, data[ind], (86 if ind != 4 else 106, i if ind != 5 else i-1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    # cv2.putText(passport, "Moosya", (86, 217), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    # cv2.putText(passport, "Cat", (86, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    # cv2.putText(passport, "2", (86, 247), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    # cv2.putText(passport, "F", (86, 262), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    #cv2.putText(passport, "12.11.2018", (106, 272), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    #cv2.putText(passport, "White&black", (86, 284), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    next_data = ["123FFC11", "25.07.2020", "Washington, USA", "7", "17.08.2020"]
    for ind, i in enumerate(range(89, 250, 40)):
        cv2.putText(passport, next_data[ind], (254, i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    # cv2.putText(passport, "123FFC11", (254, 89), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    # cv2.putText(passport, "25.07.2020", (254, 129), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    # cv2.putText(passport, "Washington, USA", (254, 169), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    # cv2.putText(passport, "7", (254, 209), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    # cv2.putText(passport, "17.08.2020", (254, 249), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    cv2.imshow("passport", passport)
    cv2.imwrite("filled_passport.jpg", passport)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
