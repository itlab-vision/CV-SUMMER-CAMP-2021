import argparse
import cv2
import sys
from random import randint, choice


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

    # Exit if nothing was detected
    if not rects.size:
        return None

    # Crop image
    x, y, w, h = rects[0]
    cat_image = image[y+2:y+h-1, x+2:x+w-1]
    return cat_image

    # Save result image to file
    # cv2.imwrite('out.jpg', image)


def make_cat_passport(cat_image, passport_image_path, output_path):

    # Read passport image
    passport_image = cv2.imread(passport_image_path)

    x1, y1 = 31, 48
    x2, y2 = 200+1, 186+1

    # Resize cat image to fit
    cat_image = cv2.resize(cat_image, (x2-x1, y2-y1), interpolation=cv2.INTER_AREA)

    # Put cat image on passport
    passport_image[y1:y2, x1:x2] = cat_image

    # Put text
    cv2.putText(
        passport_image, choice(('Max', 'Chloe', 'Bella', 'Oliver', 'Smokey', 'Lucy', 'Charlie')),
        (100, 218), cv2.FONT_HERSHEY_COMPLEX, 0.64, (10, 10, 10), 1)
    cv2.putText(
        passport_image, f'{randint(1, 30)}:{randint(1,12):02}:2020',
        (122, 272), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (10, 10, 10), 1)
    cv2.putText(
        passport_image, ''.join(choice('0123456789') for _ in range(14)),
        (275, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.49, (10, 10, 10), 1)
    cv2.putText(
        passport_image, f'{randint(1, 30)}:{randint(1,6):02}:2021',
        (275, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.49, (10, 10, 10), 1)
    cv2.putText(
        passport_image, 'Shoulder Blades',
        (275, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.49, (10, 10, 10), 1)
    cv2.putText(
        passport_image, ''.join(choice('0123456789') for _ in range(14)), 
        (275, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.49, (10, 10, 10), 1)
    cv2.putText(
        passport_image, f'{randint(1, 30)}:{randint(1,6):02}:2021',
        (275, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.49, (10, 10, 10), 1)

    # Display passport
    cv2.imshow("Cat Passport", passport_image)

    # Save passport to file
    if output_path:
        cv2.imwrite(output_path, passport_image)
        print(f'Image saved to {output_path}')



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
    args.add_argument('-o', '--output', type=str, required=False,
                      help='Passport output path')
    return parser


def main():
    
    args = build_argparser().parse_args()
    cat_image = detect_cat_in_image(args.input, args.model)
    if cat_image is None:
        print('No cat was detected in the image :(')
        return 0
    make_cat_passport(cat_image, args.passport, args.output)
    cv2.waitKey()
    cv2.destroyAllWindows()

    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
