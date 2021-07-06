import argparse
import sys
import numpy as np
import cv2


def make_cat_passport_image(input_image_path, input_passport_path, haar_model_path):

    # Read image
    image = cv2.imread(input_image_path)
    passport = cv2.imread(input_passport_path, cv2.IMREAD_UNCHANGED)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize image intensity
    gray = cv2.equalizeHist(gray)

    # Resize image
    resized = cv2.resize(gray, (640, 480), interpolation = cv2.INTER_AREA)

    # Detect cat faces using Haar Cascade
    detector = cv2.CascadeClassifier(haar_model_path)
    rects = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(75, 75))

    # Draw bounding box
    for (i, (x, y, w, h)) in enumerate(rects):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, "Cat #{}".format(i + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # Display result image
    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Crop image
    x, y, w, h = rects[0]
    image = image[y:y+h, x:x+w]

    # Add photo to passport
    new_size = (175, 145)
    image = cv2.resize(image, new_size, interpolation = cv2.INTER_CUBIC)
    rows,cols = image.shape[:2]
    passport[45:rows + 45, 30:cols + 30] = image

    # Add data to passport
    name_coord = (87, 217)
    color = (0, 0, 255)
    cv2.putText(passport, "Arseniy", name_coord, cv2.FONT_HERSHEY_PLAIN, 0.9, color, 1)

    date_coord = (111, 271)
    cv2.putText(passport, "02.08.2013", date_coord, cv2.FONT_HERSHEY_PLAIN, 0.9, color, 1)

    species_coord = (87, 233)
    cv2.putText(passport, "Dvoroviy", species_coord, cv2.FONT_HERSHEY_PLAIN, 0.9, color, 1)

    breed_coord = (87, 246)
    cv2.putText(passport, "No data", breed_coord, cv2.FONT_HERSHEY_PLAIN, 0.9, color, 1)

    sex_coord = (87, 259)
    cv2.putText(passport, "Male", sex_coord, cv2.FONT_HERSHEY_PLAIN, 0.9, color, 1)

    coat_coord = (87, 286)
    cv2.putText(passport, "White-black", coat_coord, cv2.FONT_HERSHEY_PLAIN, 0.9, color, 1)

    cv2.imshow("Result", passport)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save result image to file
    cv2.imwrite('out.jpg', passport)
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
    args.add_argument('-p', '--passport', type=str, required=True,
                      help='Required. Path to input passport')                 
    return parser


def main():
    
    args = build_argparser().parse_args()
    make_cat_passport_image(args.input, args.passport, args.model)

    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
