from google.oauth2 import service_account
from google.cloud import vision
from opencc import OpenCC
import io
import jieba

path = './images/56401.jpg'

# Below not working
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\Tibame_T14\Documents\shunzezheng\key.json"

creds = service_account.Credentials.from_service_account_file(r"key.json")


def detect_text(path):
    client = vision.ImageAnnotatorClient(credentials=creds)

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    # print('Texts:')

    for text in texts:
        # print('\n"{}"'.format(text.description))
        cc = OpenCC('s2t')
        print(str(cc.convert(text.description)))

        # vertices = (['({},{})'.format(vertex.x, vertex.y)
        #              for vertex in text.bounding_poly.vertices])

        # print('bounds: {}'.format(','.join(vertices)))
        # print('bounds: {}')
    #
    # if response.error.message:
    #     raise Exception(
    #         '{}\nFor more info on error messages, check: '
    #         'https://cloud.google.com/apis/design/errors'.format(
    #             response.error.message))


if __name__ == "__main__":
    detect_text(path)
