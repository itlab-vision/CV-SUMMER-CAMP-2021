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

    # Resize image
    resized = cv2.resize(gray, (640, 480), interpolation = cv2.INTER_CUBIC)

    # Detect cat faces using Haar Cascade
    detector = cv2.CascadeClassifier(haar_model_path)
    rects = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(75, 75))
    
    # Draw bounding box
    for (i, (x, y, w, h)) in enumerate(rects):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, "Cat #{}".format(i + 1), (x, y - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    
    # Display result image
    cv2.imshow("result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Crop image
    x, y, w, h = rects[0]
    crop = image[y:y+h, x:x+w]

    # Save result image to file
    cv2.imwrite("cat_face.jpg", crop)

    # Read passport
    pet_passport = cv2.imread("pet_passport.png")

    # Resize cat face
    crop = cv2.resize(crop, (164, 134), interpolation = cv2.INTER_CUBIC)

    # Insert cat face
    pet_passport[50 : 184, 33 : 197] = crop

    # Insert text
    cv2.putText(pet_passport, "Lucy", (85, 218), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))
    cv2.putText(pet_passport, "Siamese", (85, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))
    cv2.putText(pet_passport, "Balinese-Javanese", (85, 246), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))
    cv2.putText(pet_passport, "Male", (85, 259), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))
    cv2.putText(pet_passport, "02.11.2017", (111, 272), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))
    cv2.putText(pet_passport, "Medium", (85, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))
    cv2.putText(pet_passport, "2183-1398-3515-6531", (255, 91), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))
    cv2.putText(pet_passport, "08.01.2018", (255,130), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))
    cv2.putText(pet_passport, "Top of neck", (255, 171), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))
    cv2.putText(pet_passport, "2183-1398-3515-6531", (255, 212), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))
    cv2.putText(pet_passport, "08.01.2018", (255,252), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))

    # Show result
    cv2.imshow("result", pet_passport)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Write result

    cv2.imwrite("filled_passport.jpg", pet_passport)

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
