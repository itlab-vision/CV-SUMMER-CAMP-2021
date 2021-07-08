import argparse
import cv2
import sys


def make_cat_passport_image(input_image_path, haar_model_path):

    # Read image
    image = cv2.imread(input_image_path)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize image intensity
    gray = cv2.equalizeHist(gray)
    #gray = cv2.normalize(gray, None)

    # Resize image
    #resized = cv2.resize(gray, (640, 480), interpolation = cv2.INTER_AREA)

    # Detect cat faces using Haar Cascade
    detector = cv2.CascadeClassifier(haar_model_path)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(75, 75))
    print(rects)

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
    out = image[y:y + h, x:x + w]

    # Save result image to file
    cv2.imwrite('out.jpg', out)

    # Put cropped catface to passport
    passport = cv2.imread('pet_passport.png')
    print(passport.shape)
    passport[46:186, 50:190] = out

    # Characteristics
    cv2.putText(passport, "Barsik", (85, 215),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (130, 0, 0), 1)
    cv2.putText(passport, "Cat", (90, 232),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (130, 0, 0), 1)
    cv2.putText(passport, "Home", (90, 245),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (130, 0, 0), 1)
    cv2.putText(passport, "Male", (90, 257),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (130, 0, 0), 1)
    cv2.putText(passport, "04.05.2015", (110, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (130, 0, 0), 1)
    cv2.putText(passport, "White", (90, 282),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (130, 0, 0), 1)
    cv2.putText(passport, "83123123", (255, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (130, 0, 0), 1)
    cv2.putText(passport, "05.01.2019", (255, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (130, 0, 0), 1)
    cv2.putText(passport, "Nizhny Novgorod", (255, 169),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (130, 0, 0), 1)
    cv2.putText(passport, "88005553535", (255, 208),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (130, 0, 0), 1)
    cv2.putText(passport, "05.09.2019", (255, 246),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (130, 0, 0), 1)

    cv2.imshow("window_name", passport)
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
