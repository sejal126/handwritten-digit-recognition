import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
from hand import HandwrittenDigitRecognizer

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a Digit")
        
        # Create canvas
        self.canvas = tk.Canvas(root, width=280, height=280, bg='white')
        self.canvas.pack()
        
        # Create buttons
        self.clear_button = tk.Button(root, text="Clear", command=self.clear)
        self.clear_button.pack(side=tk.LEFT)
        
        self.predict_button = tk.Button(root, text="Predict", command=self.predict)
        self.predict_button.pack(side=tk.LEFT)
        
        self.result_label = tk.Label(root, text="")
        self.result_label.pack()
        
        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.draw)
        
        # Create image for saving
        self.image = Image.new("L", (280, 280), "white")
        self.image_draw = ImageDraw.Draw(self.image)
        
    def draw(self, event):
        x, y = event.x, event.y
        # Draw a thick line
        self.canvas.create_oval(x-10, y-10, x+10, y+10, fill='black')
        self.image_draw.ellipse([x-10, y-10, x+10, y+10], fill='black')
        
    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="")
        
    def predict(self):
        # Resize image to 28x28
        resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Save image
        resized.save("test_digit.png")
        
        # Create recognizer and predict
        recognizer = HandwrittenDigitRecognizer()
        
        # Make prediction
        digit, predictions = recognizer.predict("test_digit.png")
        
        # Format predictions
        predictions_text = "\n".join(predictions)
        self.result_label.config(text=f"Top predictions:\n{predictions_text}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
