# Handwritten Digit Recognition Dashboard

A modern GUI application that uses a neural network to recognize handwritten digits.

## Features

- Real-time digit drawing
- Neural network-based digit recognition
- Clean and modern dashboard interface
- Prediction confidence display
- Top predictions statistics

## Requirements

- Python 3.8+
- TensorFlow
- Pillow (PIL)
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python digit_dashboard.py
```

Draw a digit in the left panel and click "Predict Digit" to see the results in the right panel.

## Project Structure

- `digit_dashboard.py`: Main application file with the dashboard interface
- `draw_digit.py`: Drawing functionality implementation
- `hand.py`: Neural network model and prediction implementation
- `requirements.txt`: Project dependencies
