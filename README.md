# Ingredient Image Classification and AI Pantry

This project aims to build a web application that can classify ingredients from images, maintain an inventory of ingredients ("AI Pantry"), and provide recipe recommendations.

The repository currently includes:

- `Pic_n_Dine.py` – the original Colab notebook script used for training a ResNet‑34 model on ingredient images.
- `main.py` – Flask API providing ingredient classification, inventory management, and recipe recommendation.
- `train.py` – script to train a ResNet‑34 ingredient classifier given a dataset split.
- `recipes.json` – simple recipe database used for recommendations.
- `inventory.json` – persistent storage for the inventory.
- `requirements.txt` – Python dependencies for running the web app.

## Planned Tasks

1. **Dataset Preparation**
   - Organize the APS360 ingredient dataset locally.
   - Script the train/validation/test split as used in the original notebook.

2. **Model Training**
   - Refactor `Pic_n_Dine.py` into a reusable training script.
   - Train a ResNet‑34 classifier and export the model weights.

3. **API Development**
   - Replace the dummy model in `main.py` with the trained model.
   - Implement an endpoint to recommend recipes based on current inventory.

4. **Web Interface**
   - Create a simple frontend to upload images, view inventory, adjust quantities, and view recipe suggestions.

5. **Inventory Management**
   - Improve persistence (e.g., SQLite) and add functionality to decrease quantities when recipes are cooked.

6. **Testing & Deployment**
   - Add unit tests for API routes and model inference.
   - Provide Dockerfile and deployment instructions.

## Running the Web App

Install dependencies and start the Flask server:

```bash
pip install -r requirements.txt
python main.py
```

The API will be available at `http://localhost:5000/`.

## Training the Model

If you have the ingredient dataset split into `train`, `val`, and `test` directories,
you can train the classifier with:

```bash
python train.py --data /path/to/dataset_split --output model.pt
```

Place the resulting `model.pt` file in the repository root so `main.py` will load it
automatically.
