# 🎨 Doodle Detect

A machine learning project that recognizes hand-drawn doodles using a Convolutional Neural Network (CNN). Draw something on the canvas, and watch the AI guess what it is!

## 🚀 Features

- **Interactive Drawing Canvas**: Draw doodles with your mouse
- **Real-time Prediction**: AI analyzes your drawings instantly
- **42 Different Classes**: Recognizes objects like animals, food, vehicles, and more
- **Confidence Scoring**: Shows how confident the AI is about its predictions
- **Top 3 Predictions**: See alternative guesses for better insights
- **Image Preprocessing Visualization**: View how your drawing is processed

## 🎯 Supported Objects

The model can recognize 42 different objects:

- **Animals**: ant, bird, cat, dog, duck, fish, octopus
- **Food**: apple, banana, cookie, donut, hamburger, ice cream, lollipop, strawberry, watermelon
- **Vehicles**: bicycle, car
- **Sports**: basketball, skateboard
- **Body Parts**: arm, ear, eye, hand, nose
- **Objects**: axe, bat, book, camera, computer, door, guitar, microphone, radio, sword, t-shirt, television, tree, violin, wristwatch
- **Landmarks**: The Eiffel Tower
- **Abstract**: rainbow

## 📁 Project Structure

```
doodledetect/
├── app.py                 # Streamlit app
├── train.py              # Clean training script
├── doodle_dataset.py     # Dataset loader
├── doodle_model.pt       # Trained model (from train.py)
├── requirements.txt      # Python dependencies
└── README.md
```

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- pip

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/UltimateHobbyCoder/doodledetect.git
   cd doodledetect
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv doodledetect
   source doodledetect/bin/activate  # On Windows: doodledetect\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Usage

### Training the Model

1. **Prepare your data**: Place your `.npy` files in a `data/` directory
2. **Train the model**:
   ```bash
   python train.py
   ```
   This will create `doodle_model.pt` with your trained model.

### Running the Interactive App

1. **Start the Streamlit app**:
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Open your browser** and go to `http://localhost:8501`

3. **Draw and predict**:
   - Draw something on the canvas
   - Click "Predict Drawing"
   - See what the AI thinks you drew!

## 🧠 Model Architecture

The CNN model consists of:
- **Input Layer**: 28x28 grayscale images
- **Convolutional Layers**: 2 layers with ReLU activation and max pooling
- **Fully Connected Layers**: 2 dense layers for classification
- **Output**: 42-class softmax classification

```
Input (1x28x28) → Conv2d(32) → Pool → Conv2d(64) → Pool → FC(128) → FC(42)
```

## 📊 Model Performance

- **Training Data**: 5,000 samples per class (210,000 total images)
- **Training/Validation Split**: 80/20
- **Batch Size**: 64
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: CrossEntropyLoss

## 🎨 How It Works

1. **Data Loading**: QuickDraw dataset files are loaded as 28x28 numpy arrays
2. **Preprocessing**: Images are converted to tensors and normalized
3. **Training**: CNN learns to classify 42 different doodle categories
4. **Inference**: User drawings are preprocessed and fed to the trained model
5. **Prediction**: Model outputs class probabilities and top predictions

## 🔧 Troubleshooting

### Common Issues

**Model keeps predicting the same class:**
- Check that class names are in the correct alphabetical order
- Verify the model file exists and loads correctly
- Ensure preprocessing matches training data format

**Poor prediction accuracy:**
- Try drawing more clearly and centered
- Make sure your drawing resembles the training data style
- Check if the object is in the supported classes list

**Canvas not working:**
- Ensure `streamlit-drawable-canvas` is installed
- Try refreshing the browser page
- Check browser compatibility (works best in Chrome/Firefox)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Google QuickDraw Dataset**: For providing the training data
- **PyTorch**: For the deep learning framework
- **Streamlit**: For the interactive web interface
- **Streamlit Drawable Canvas**: For the drawing functionality

## 📞 Support

If you have any questions or issues: Open an issue on GitHub
