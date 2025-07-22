import json
import os
from pathlib import Path
from typing import Dict

from flask import Flask, jsonify, request
from PIL import Image
import torch
import torchvision.transforms as transforms

app = Flask(__name__)

# Placeholder model - to be replaced with a real trained model
class DummyModel:
    classes = [
        'apple', 'asparagus', 'avocado', 'banana', 'bean', 'beetroot',
        'bell pepper', 'bread', 'broccoli', 'butter', 'cabbage', 'carrot',
        'cauliflower', 'cucumber', 'grapes', 'lemon', 'lettuce', 'meatloaf',
        'onion', 'potato', 'pumpkin', 'radish', 'raw_beef', 'raw_chicken',
        'raw_pork', 'red_wine', 'salmon', 'spinach', 'sweet_potato',
        'tomato', 'zucchini'
    ]

    def __call__(self, img_tensor: torch.Tensor) -> int:
        # Dummy implementation always returns 0 ("apple")
        return 0

model = DummyModel()

def load_inventory(path: Path) -> Dict[str, int]:
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return {}

def save_inventory(path: Path, inventory: Dict[str, int]):
    with open(path, 'w') as f:
        json.dump(inventory, f)

inventory_path = Path('inventory.json')

@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    img_file = request.files['image']
    img = Image.open(img_file.stream).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        pred_idx = model(img_tensor)
    predicted = DummyModel.classes[pred_idx]
    return jsonify({'ingredient': predicted})

@app.route('/inventory', methods=['GET'])
def get_inventory():
    inv = load_inventory(inventory_path)
    return jsonify(inv)

@app.route('/inventory/add', methods=['POST'])
def add_inventory():
    data = request.get_json(force=True)
    name = data.get('name')
    qty = int(data.get('quantity', 1))
    if not name:
        return jsonify({'error': 'name required'}), 400
    inv = load_inventory(inventory_path)
    inv[name] = inv.get(name, 0) + qty
    save_inventory(inventory_path, inv)
    return jsonify(inv)

@app.route('/inventory/reduce', methods=['POST'])
def reduce_inventory():
    data = request.get_json(force=True)
    name = data.get('name')
    qty = int(data.get('quantity', 1))
    if not name:
        return jsonify({'error': 'name required'}), 400
    inv = load_inventory(inventory_path)
    if name in inv:
        inv[name] = max(inv[name] - qty, 0)
        save_inventory(inventory_path, inv)
    return jsonify(inv)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
