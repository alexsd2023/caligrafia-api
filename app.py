#!/usr/bin/env python3
import os
from flask import Flask, request, jsonify
from clasificador_CNN import CaligrafiaPredictor
from PIL import Image
import torch
import io

app = Flask(__name__)

# Cargar predictor al inicio
print("🔥 Cargando modelo...")
predictor = CaligrafiaPredictor("best_caligrafia_model.pth")
print("✅ Modelo listo!")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "OK", 
        "device": str(predictor.device),
        "model_loaded": True
    })

@app.route('/predict_caligrafia', methods=['POST'])
def predict_caligrafia():
    if 'image' not in request.files:
        return jsonify({"error": "Falta imagen"}), 400
    
    file = request.files['image']
    try:
        # Procesar imagen directamente desde bytes
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        resultado = predictor.predict_image(image)  # Modifica tu clase para esto
        
        return jsonify(resultado)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "api": "Caligrafía CRNN API",
        "endpoints": ["/health", "/predict_caligrafia"],
        "status": "🚀 Listo para predicciones"
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
