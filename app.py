# app.py for Render-deployable Flask Server with Two Models (cat + dog)

from flask import Flask, request, jsonify
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import MobileNetClassifier

app = Flask(__name__)

# Load both models
cat_model = MobileNetClassifier(num_classes=4)
cat_model.load_state_dict(torch.load("cat_multi_mobilenet.pth", map_location="cpu"))
cat_model.eval()

dog_model = MobileNetClassifier(num_classes=4)
dog_model.load_state_dict(torch.load("dog_multi_mobilenet.pth", map_location="cpu"))
dog_model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files or 'species' not in request.form:
        return jsonify({"error": "Both 'image' and 'species' fields are required."}), 400

    image_file = request.files['image']
    species = request.form['species'].lower()

    if species not in ['cat', 'dog']:
        return jsonify({'error': "species must be 'cat' or 'dog'"}), 400

    image = Image.open(image_file).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    model = cat_model if species == 'cat' else dog_model

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)
        pred = torch.argmax(prob, dim=1).item()
        confidence = prob[0][pred].item()

    label = _convert_label(pred, species)

    return jsonify({
        'species': species,
        'label': label,
        'confidence': round(confidence, 4)
    })

def _convert_label(index, species):
    label_map_cat = {
        0: '정상',
        1: '각막염',
        2: '결막염',
        3: '유루증'
    }
    label_map_dog = {
        0: '정상',
        1: '안검염',
        2: '백내장',
        3: '녹내장'
    }
    return (label_map_cat if species == 'cat' else label_map_dog).get(index, '알 수 없음')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
