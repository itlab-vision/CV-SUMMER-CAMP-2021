import sys
import argparse
from random import randint
import cv2


def crop_image(image, x, y, w, h):
    return image[y:y + h, x:x + w]


def resize_image(image, new_width, new_height):
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def create_passport(passport, photo, cat_index):

    # Put cat image
    filled_passport = passport.copy()
    photo = resize_image(photo, 176, 144)
    filled_passport[45:photo.shape[0] + 45, 28:photo.shape[1] + 28] = photo

    # Put other cat data
    gender = ["Male", "Female"]
    breeds = ["Chinchilla", "Aegean", "American Shorthair", "Asian"]

    cv2.putText(filled_passport, "Cat #{}".format(cat_index + 1), (89, 218), cv2.FONT_ITALIC, 0.55, (0, 0, 0), 1)
    cv2.putText(filled_passport, breeds[randint(0, 3)], (89, 247), cv2.FONT_ITALIC, 0.55, (0, 0, 0), 1)
    cv2.putText(filled_passport, gender[randint(0, 1)], (89, 260), cv2.FONT_ITALIC, 0.55, (0, 0, 0), 1)
    cv2.putText(filled_passport, f"{randint(1, 30)}.{randint(1, 12)}.20{randint(10, 20)}", (105, 273), cv2.FONT_ITALIC, 0.55, (0, 0, 0), 1)
    cv2.putText(filled_passport, "1337", (256, 93), cv2.FONT_ITALIC, 0.55, (0, 0, 0), 1)

    cv2.imshow("passport", filled_passport)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(f"filled_passport_{cat_index}.jpg", filled_passport)


def make_cat_passport_image(input_image_path, haar_model_path):

    # Read image
    image = cv2.imread(input_image_path)
    passport = cv2.imread("pet_passport.png")

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize image intensity
    gray = cv2.equalizeHist(gray)

    # Detect cat faces using Haar Cascade
    detector = cv2.CascadeClassifier(haar_model_path)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(75, 75))

    # Draw bounding boxes and create passports
    for i, (x, y, w, h) in enumerate(rects):
        create_passport(passport, crop_image(image, x, y, w, h), i)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, "Cat #{}".format(i + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # Display result image
    cv2.imshow("cats", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save result image to file
    cv2.imwrite('Detected_cats.jpg', image)

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
