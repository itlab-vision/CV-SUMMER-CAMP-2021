import argparse
import sys
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
    rects = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5,
    minSize=(75, 75))


    # Draw bounding box
    for (i, (x, y, w, h)) in enumerate(rects):
        cv2.rectangle(image, (x, y), (x + w, y + h), (50, 255, 50), 2)
        cv2.putText(image, "Cat detected", (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 255, 50), 2)

    # Display result image
    cv2.imshow("window_name", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Crop image
    x, y, w, h = rects[0]
    image = image[y:y+h, x:x+w]

    # Save result image to file
    cv2.imwrite('out.jpg', image)

#-------------------------------- Extra functins----------------------------------------------------------

    # Resize result image
   # resized = cv2.resize(image, (170, 140), interpolation = cv2.INTER_AREA)

    # Filling out the passport with data
    passport = cv2.imread('pet_passport.png')

    cv2.putText(passport, "Barsik", (88, 217),
    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    cv2.putText(passport, "Cat", (88, 230),
    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    cv2.putText(passport, "Siamese", (88, 245),
    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    cv2.putText(passport, "Male", (88, 259),
    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    cv2.putText(passport, "06.07.2019", (114, 271),
    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    cv2.putText(passport, "Short", (88, 285),
    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    cv2.putText(passport, "11111", (258, 88),
    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    cv2.putText(passport, "06.11.2019", (258, 128),
    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    cv2.putText(passport, "Paw", (258, 168),
    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    cv2.putText(passport, "1111", (258, 208),
    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    cv2.putText(passport, "06.11.2019", (258, 248),
    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    rows, cols, chanels = image.shape
    image = cv2.addWeighted(passport[45:45+rows, 50:50+cols], 0, image, 0.5, 0)
    passport[45:45+rows, 50:50+cols] = image

    

    # Save passport image to file
    cv2.imwrite('catpassport.jpg', passport)


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
