
# 🚦 CS50AI - Project 5: Traffic Sign Classification

This project is part of Harvard's CS50 Introduction to Artificial Intelligence with Python. It trains a convolutional neural network (CNN) using TensorFlow to recognize traffic signs from the [GTSRB dataset](https://benchmark.ini.rub.de/gtsrb_news.html).

---

## 📁 Dataset

The dataset contains 43 categories of German traffic signs. Each category is stored in a separate folder named from `0` to `42`.

To train the model, all images are resized to **30x30** pixels and loaded into memory using OpenCV.

---

## 🧠 Model Architecture

The CNN is built using `tensorflow.keras` and includes the following layers:

- `Conv2D` with 32 filters, 3x3 kernel, ReLU activation
- `MaxPooling2D`
- `Conv2D` with 64 filters, 3x3 kernel, ReLU activation
- `MaxPooling2D`
- `Flatten`
- `Dense` layer with 128 units, ReLU activation
- `Dropout` (rate: 0.5) to reduce overfitting
- `Dense` output layer with 43 units and `softmax` activation

Compiled with:
```python
optimizer='adam', 
loss='categorical_crossentropy',
metrics=['accuracy']
```

---

## 🚀 How to Run

1. **Install dependencies** (inside a virtual environment):
```bash
pip install -r requirements.txt
```

2. **Download and extract the GTSRB dataset** into a folder named `gtsrb`.

3. **Train the model**:
```bash
python traffic.py gtsrb model.h5
```

This will train the CNN and save the model to `model.h5`.

---

## 📊 Performance

After 10 epochs of training, the model achieved:

- **Training Accuracy**: 85.96%
- **Validation Accuracy**: **96.37%**

---

## 💡 Notes

- If you don't have a GPU or CUDA drivers, TensorFlow will run on CPU (no problem).
- Consider saving your model in the new `.keras` format as recommended by Keras.

---

## 👤 Author

Project completed by [Your Name Here] as part of CS50AI.

---
