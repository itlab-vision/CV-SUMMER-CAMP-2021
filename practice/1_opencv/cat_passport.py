import sys
import argparse
import cv2


def make_cat_image(input_image_path, haar_model_path):

    
    image = cv2.imread(input_image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    detector = cv2.CascadeClassifier(haar_model_path)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(75, 75))

    for (i, (x, y, w, h)) in enumerate(rects):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, "Cat #{}".format(i + 1), (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    x, y, w, h = rects[0]
    image = image[y:y+h, x:x+w]

    PH_WIDTH, PH_HEIGHT = 180, 150
    resized = cv2.resize(image, (PH_WIDTH, PH_HEIGHT), interpolation = cv2.INTER_CUBIC)


    return resized

def make_cat_image_passport(cat_img, passport_path):
    X_COORD, Y_COORD = 25, 45
    
    passport = cv2.imread(passport_path)
    passport[ Y_COORD:Y_COORD+cat_img.shape[0], X_COORD:X_COORD+cat_img.shape[1]] = cat_img

    data = ["Name", "Specie", "Breed", "Sex", "dd.mm.yyyy", "Coat"]
    i = 0

    for _ in range(215, 215+15*(len(data)), 15):
        cv2.putText(passport, data[i], (115, _), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (0,0,0), 1)
        i+=1

    cv2.imwrite('passport.jpg', passport)
    return passport

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
                      help='Required. Path to passport image')
    return parser


def main():
    
    args = build_argparser().parse_args()
    img = make_cat_image(args.input, args.model)
    passport =make_cat_image_passport(img, args.passport)
    cv2.imshow("window_name", passport)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
