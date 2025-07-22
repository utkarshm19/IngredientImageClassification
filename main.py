import json
import os
from pathlib import Path
from typing import Dict, List

from flask import Flask, jsonify, request
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn

app = Flask(__name__)

# Placeholder model - used if no trained model is available
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

def build_resnet(num_classes: int):
    model = torchvision.models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.eval()
    return model


def load_model(model_path: Path):
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location='cpu')
        classes = checkpoint['classes']
        m = build_resnet(len(classes))
        m.load_state_dict(checkpoint['model_state'])
        return m, classes
    return DummyModel(), DummyModel.classes


model_path = Path('model.pt')
model, model_classes = load_model(model_path)

def load_inventory(path: Path) -> Dict[str, int]:
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return {}

def save_inventory(path: Path, inventory: Dict[str, int]):
    with open(path, 'w') as f:
        json.dump(inventory, f)


def load_recipes(path: Path) -> List[Dict]:
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return []

inventory_path = Path('inventory.json')
recipes_path = Path('recipes.json')

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
        if isinstance(model, DummyModel):
            pred_idx = model(img_tensor)
        else:
            outputs = model(img_tensor)
            pred_idx = outputs.argmax(dim=1).item()
    predicted = model_classes[pred_idx]
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


def recommend_recipes(inv: Dict[str, int], recipes: List[Dict]) -> List[Dict]:
    available = []
    for r in recipes:
        ok = True
        for ing, qty in r.get('ingredients', {}).items():
            if inv.get(ing, 0) < qty:
                ok = False
                break
        if ok:
            available.append(r)
    return available


@app.route('/recipes', methods=['GET'])
def recipes():
    inv = load_inventory(inventory_path)
    recs = load_recipes(recipes_path)
    available = recommend_recipes(inv, recs)
    return jsonify(available)


@app.route('/recipes/cook', methods=['POST'])
def cook_recipe():
    data = request.get_json(force=True)
    recipe_name = data.get('name')
    recipes = load_recipes(recipes_path)
    recipe = next((r for r in recipes if r['name'] == recipe_name), None)
    if not recipe:
        return jsonify({'error': 'recipe not found'}), 404

    inv = load_inventory(inventory_path)
    for ing, qty in recipe['ingredients'].items():
        if inv.get(ing, 0) < qty:
            return jsonify({'error': 'not enough ingredients'}), 400
    for ing, qty in recipe['ingredients'].items():
        inv[ing] = inv.get(ing, 0) - qty
    save_inventory(inventory_path, inv)
    return jsonify(inv)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
