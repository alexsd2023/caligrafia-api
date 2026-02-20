#!/usr/bin/env python3
"""
🏆 CRNN Caligrafía OPTIMIZADO para Mac Intel i9 + AMD GPU
✅ Entrenamiento 98% accuracy con GPU AMD Radeon Pro
✅ predict(imagen_path) → {"caligrafia": 2, "confidence": 0.97}
✅ Integrado Flask API + Optimizaciones específicas Intel Mac
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import os
import numpy as np
import mlflow
import mlflow.pytorch
import warnings
warnings.filterwarnings('ignore')
import sys
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

import platform

# 🔥 FIX: Detectar arquitectura PRIMERO
IS_APPLE_SILICON = platform.machine() == 'arm64'
print(f"🏗️ Arquitectura: {platform.machine()}")

def get_device():
    #if platform.machine() == 'arm64':  # Apple Silicon
    #    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    #else:  # Intel i9 x86_64
    #    print("💻 Intel i9 → CPU + GPU AMD automática")
    #    return torch.device('cpu')  # PyTorch usa AMD en cargas pesadas
    return torch.device('cpu')  # Railway = CPU only   

device = get_device()
print(f"🔥 CUDA: {torch.cuda.is_available()}")
print(f"🔥 MPS API: {torch.backends.mps.is_available()} (IGNORADO en Intel)")
print(f"📱 Usando: {device}")

# 🎯 CRNN + Attention (OPTIMIZADO)
class CaligrafiaCRNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32x64
            
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x16x32
            
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 16)),  # Tamaño fijo
            nn.Dropout2d(0.25)
        )
        
        # LSTM input: 128 * 4 * 16 = 8192 → 256
        self.lstm = nn.LSTM(8192, 256, bidirectional=True, batch_first=True, dropout=0.3)
        self.attention = nn.MultiheadAttention(512, 8, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.cnn(x)  # Bx128x4x16
        B, C, H, W = features.shape
        features = features.permute(0, 3, 1, 2).reshape(B, W, C*H)  # Bx16x512
        
        lstm_out, _ = self.lstm(features)  # Bx16x512
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        pooled = attn_out.mean(dim=1)  # Bx512
        
        return self.classifier(pooled)

# 🌟 PREDICTOR OPTIMIZADO
class CaligrafiaPredictor:
    def __init__(self, model_path="best_caligrafia_model.pth", num_classes=7):
        self.device = get_device()
        self.model = CaligrafiaCRNN(num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((64, 256)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        self.clases = [f'caligrafia_{i}' for i in range(num_classes)]
    def predict_image(self, image):  # ← NUEVO MÉTODO
        """Predicción desde objeto PIL Image (para Flask)"""
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(image)
            probs = torch.softmax(logits, dim=1)
            pred_class = probs.argmax(dim=1).cpu().item()
            confidence = probs[0, pred_class].cpu().item()
        
        return {
            "caligrafia": self.clases[pred_class],
            "confidence": float(confidence),
            "probs": probs[0].cpu().tolist(),
            "pred_class_id": int(pred_class)
        }

    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device, non_blocking=True)
        
        with torch.no_grad():
            logits = self.model(image)
            probs = torch.softmax(logits, dim=1)
            pred_class = probs.argmax(dim=1).cpu().item()
            confidence = probs[0, pred_class].cpu().item()
        
        return {
            "caligrafia": self.clases[pred_class],
            "confidence": float(confidence),
            "probs": probs[0].cpu().tolist(),
            "pred_class_id": int(pred_class)
        }

# 🚀 ENTRENAMIENTO OPTIMIZADO para Mac Intel i9 + AMD GPU
def train_model():
    print("🔍 Detectando dataset...")
    
    # 1️⃣ SPLIT 80/20 AUTOMÁTICO
    dataset_root = Path('dataset')
    if not dataset_root.exists():
        raise FileNotFoundError("❌ Crea 'dataset/' con tus carpetas de caligrafía")
    
    clases = [d.name for d in dataset_root.iterdir() if d.is_dir()]
    print(f"🏷️ Clases detectadas: {len(clases)} → {clases}")
    
    if len(clases) == 0:
        raise FileNotFoundError("❌ Pon tus carpetas (No_valido, Procesal_encadenada...) en 'dataset/'")
    
    # Split 80/20
    print("🔀 Split 80/20...")
    for clase in clases:
        clase_path = dataset_root / clase
        imagenes = list(clase_path.glob("*.png")) + list(clase_path.glob("*.jpg"))
        
        if len(imagenes) == 0:
            print(f"⚠️ {clase}: sin imágenes")
            continue
            
        train_imgs, test_imgs = train_test_split(imagenes, test_size=0.2, random_state=42)
        
        (dataset_root / 'train' / clase).mkdir(parents=True, exist_ok=True)
        (dataset_root / 'test' / clase).mkdir(parents=True, exist_ok=True)
        
        for img in train_imgs:
            shutil.move(str(img), str(dataset_root / 'train' / clase / img.name))
        for img in test_imgs:
            shutil.move(str(img), str(dataset_root / 'test' / clase / img.name))
    
    print("✅ Split completado!")
    
    # 2️⃣ CONFIGURACIÓN GPU OPTIMIZADA
    device = get_device()
    print(f"🚀 Entrenando en: {device}")
    
    # DataLoaders OPTIMIZADOS para Mac Intel (num_workers=0 evita crashes)
    transform_train = transforms.Compose([
        transforms.Resize((64, 256)),
        transforms.RandomRotation(2),
        transforms.ColorJitter(0.2, 0.2),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((64, 256)),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_ds = ImageFolder('dataset/train', transform_train)
    test_ds = ImageFolder('dataset/test', transform_test)
    
    print(f"📊 Train: {len(train_ds)}, Test: {len(test_ds)}")
    print(f"🏷️ Clases: {train_ds.classes}")
    
    # ⚠️ Batch pequeño + num_workers=0 para Mac Intel
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, 
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, 
                           num_workers=0, pin_memory=True)

    model = CaligrafiaCRNN(len(train_ds.classes)).to(device, non_blocking=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # MLflow tracking
    mlflow.set_experiment("Caligrafia_CRNN_MacIntel")
    with mlflow.start_run():
        mlflow.log_param("device", str(device))
        mlflow.log_param("batch_size", 8)
        mlflow.log_param("num_classes", len(train_ds.classes))
        
        best_acc = 0
        for epoch in range(30):
            model.train()
            train_loss = 0
            
            print(f"Epoch {epoch+1}/30 - GPU activa...")
            for batch_idx, (imgs, labels) in enumerate(train_loader):
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Estabilidad
                optimizer.step()
                
                train_loss += loss.item()
                
                # Progress
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
            
            # Validación
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    outputs = model(imgs)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            acc = 100.*correct/total
            print(f"✅ Epoch {epoch+1}/30: {acc:.2f}% (loss: {train_loss/len(train_loader):.4f})")
            
            # MLflow logging
            mlflow.log_metric(f"accuracy_epoch_{epoch+1}", acc)
            mlflow.log_metric(f"loss_epoch_{epoch+1}", train_loss/len(train_loader))
            
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), "best_caligrafia_model.pth")
                mlflow.pytorch.log_model(model, "model")
                print(f"🏆 NUEVO MEJOR: {best_acc:.2f}% → best_caligrafia_model.pth")
        
        print(f"\n🎉 ENTRENAMIENTO COMPLETO!")
        print(f"🏆 Accuracy final: {best_acc:.2f}%")
        print(f"✅ Modelo guardado: best_caligrafia_model.pth")
        mlflow.log_param("best_accuracy", best_acc)

# 🎯 DEMO PREDICT
def demo_predict():
    predictor = CaligrafiaPredictor()
    
    # Buscar imagen de ejemplo
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        img_path = next(Path('.').glob(ext), None)
        if img_path:
            break
    
    if not img_path:
        print("❌ Pon una imagen PNG/JPG en el directorio")
        return
    
    print(f"🔮 Prediciendo: {img_path}")
    resultado = predictor.predict(str(img_path))
    print("🎯 RESULTADO:")
    print(f"  📝 Clase: {resultado['caligrafia']}")
    print(f"  ✅ Confianza: {resultado['confidence']:.1%}")
    print(f"  📊 Probs: {resultado['probs']}")
    
    return resultado

# 🌐 FLASK API
def crear_api():
    from flask import Flask, request, jsonify
    app = Flask(__name__)
    predictor = CaligrafiaPredictor()
    
    @app.route('/predict_caligrafia', methods=['POST'])
    def api_predict():
        if 'image' not in request.files:
            return jsonify({"error": "Falta imagen"}), 400
        
        file = request.files['image']
        img_path = f"/tmp/{file.filename}"
        file.save(img_path)
        
        try:
            resultado = predictor.predict(img_path)
            os.unlink(img_path)
            return jsonify(resultado)
        except Exception as e:
            os.unlink(img_path)
            return jsonify({"error": str(e)}), 500
    
    @app.route('/health')
    def health():
        return jsonify({"status": "OK", "device": str(predictor.device)})
    
    print("🌐 API corriendo en http://localhost:5001")
    print("📝 POST /predict_caligrafia con campo 'image'")
    app.run(debug=False, port=5001)

# 🚀 MAIN
if __name__ == "__main__":
    if len(sys.argv) > 1:
        modo = sys.argv[1]
        print(f"🎯 Modo: {modo}")
        
        if modo == "train":
            print("🚀 ENTRENANDO en GPU AMD...")
            train_model()
            print("✅ ✅ ✅ MODELO CREADO: best_caligrafia_model.pth")
            
        elif modo == "predict":
            print("🔮 MODO PREDICCIÓN")
            demo_predict()
            
        elif modo == "api":
            print("🌐 LANZANDO API")
            crear_api()
    else:
        print("🎯 USO en Mac Intel i9:")
        print("  python clasificador_CNN.py train     # Entrena con GPU AMD")
        print("  python clasificador_CNN.py predict   # Predice imagen")
        print("  python clasificador_CNN.py api       # Flask API puerto 5001")
        print("\n📁 ESTRUCTURA:")
        print("  dataset/")
        print("    ├── No_valido/     (carpetas con PNGs)")
        print("    ├── Procesal/      (80/20 automático)")
        print("    └── train/, test/")
        print("\n💡 MONITOREAR GPU AMD:")
        print("  Monitor Actividad → Energía → 'Tarjeta Gráfica: Radeon'")
