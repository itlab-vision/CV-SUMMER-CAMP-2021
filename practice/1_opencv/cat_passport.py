import sys
import argparse
import cv2


def make_cat_passport_image(input_image_path, haar_model_path):

    # Read image

    image = cv2.imread(input_image_path)

    # Convert image to grayscale

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize image intensity

    gray = cv2.equalizeHist(gray)

    # Resize image

    # Detect cat faces using Haar Cascade

    detector = cv2.CascadeClassifier(haar_model_path)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(75, 75))

    # Draw bounding box

    for (i, (x, y, w, h)) in enumerate(rects):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, "Cat #{}".format(i + 1), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # Display result image

    cv2.imshow("window_name", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Crop image

    x, y, w, h = rects[0]
    image = image[y:y + h, x:x + w]

    # Save result image to file

    cv2.imwrite('out.jpg', image)

    image = cv2.imread('pet_passport.png')
    src = cv2.imread('out.jpg')
    image[47:47+src.shape[0], 45:45+src.shape[1]] = src

    cv2.putText(image, "Vasya", (88, 217), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(image, "Maine Coon", (88, 231), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(image, "Normal", (88, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(image, "Male", (88, 258), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(image, "07.04.2019", (113, 272), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(image, "Unknown", (88, 284), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    cv2.putText(image, "88005553535", (300, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(image, "07.04.2019", (300, 129), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(image, "Arizona", (300, 169), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(image, "4826", (300, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(image, "11.06.2019", (300, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    cv2.imshow("window_name", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('pet_passport.png', image)
    print("I saved everything!")
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
