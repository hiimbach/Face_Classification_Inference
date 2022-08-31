from PIL import Image
from infer import Face_Classification_Model

img = Image.open('sample_data/sample.png')


model = Face_Classification_Model()
result = model.predict(img)
