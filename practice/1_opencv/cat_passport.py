import argparse
import cv2
import sys

def show(win_name, image):
    cv2.imshow(win_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def make_cat_passport_image(input_image_path, haar_model_path):

    # Read image
    image = cv2.imread(input_image_path)
    # show("BGR Cat", image)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Normalize image intensity
    gray = cv2.equalizeHist(gray)
    # show("Gray Cat", gray)

    # Resize image
    resized = cv2.resize(gray, (640, 480), interpolation=cv2.INTER_AREA)
     # show("Resized gray Cat", resized)

    # Detect cat faces using Haar Cascade
    detector = cv2.CascadeClassifier(haar_model_path)
    rects = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(75, 75))
    
    # Draw bounding box
    for i, (x, y, w, h) in enumerate(rects):
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, "Cat #{}".format(i + 1), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # Display result image
    show("Detected Cat", image)

    # Crop image
    x, y, w, h = rects[0]
    crop_image = image[y:y+h, x:x+w]
    # show("Crop Cat", crop_image)

    # Save result image to file
    cv2.imwrite("out.jpg", crop_image)
    return crop_image

def create_cat_passport(crop_image):
    (x1, y1), (x2, y2) = (28, 46), (203, 188)
    passport = cv2.imread("pet_passport.png")
    # show("passport", passport)

    passport[y1:y2, x1:x2] = cv2.resize(crop_image, (x2-x1, y2-y1), interpolation=cv2.INTER_CUBIC)
    # show("passport", passport)

    data = ["Cezar", "Cat", "", "Male", "07.05.2021", "B&W"]
    (x, y), dy = (130, 220), 13
    
    for i, el in enumerate(data):
        cv2.putText(passport, el, (x, y + dy * i), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))

    show("passport", passport)
    cv2.imwrite("cat_passport.jpg", crop_image)
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
    crop_image = make_cat_passport_image(args.input, args.model)
    create_cat_passport(crop_image)

    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
