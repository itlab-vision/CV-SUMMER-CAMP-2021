import argparse
import cv2
import sys


def detect_cat_in_image(input_image_path, haar_model_path):

    # Read image
    image = cv2.imread(input_image_path)

    # Resize image
    image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize image intensity
    gray = cv2.equalizeHist(gray)

    # Detect cat faces using Haar Cascade
    detector = cv2.CascadeClassifier(haar_model_path)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(75, 75))

    # Draw bounding box
    for (i, (x, y, w, h)) in enumerate(rects):
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(
            image, f'Cat #{i+1}', (x, y-10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # Display result image
    cv2.imshow("Cat Decetor", image)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # Exit if nothing was detected
    if not rects.size:
        return None

    # Crop image
    x, y, w, h = rects[0]
    cat_image = image[y+1:y+h-1, x+1:x+w-1]
    return cat_image

    # Save result image to file
    # cv2.imwrite('out.jpg', image)


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
    cat_image = detect_cat_in_image(args.input, args.model)
    if cat_image is None:
        print('No cat was detected in the image :(')
        return 0

    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
