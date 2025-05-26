import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import hand
import os
import io
from tkinter import messagebox

class DigitDrawer:
    def __init__(self, window=None):
        if window:
            # Dashboard mode
            self.window = window
            
            # Create canvas
            self.canvas = tk.Canvas(
                self.window, 
                width=280, 
                height=280, 
                bg='white', 
                highlightthickness=1, 
                highlightbackground="black",
                cursor="pencil"
            )
            self.canvas.pack(pady=10)
            
            # Set canvas properties
            self.canvas.config(
                insertwidth=0,
                insertborderwidth=0,
                insertbackground='black'
            )
            
            # Initialize drawing state
            self.drawing = False
            self.last_x = None
            self.last_y = None
            self.points = []  # List to store points for smoothing
            
            # Create recognizer
            try:
                self.recognizer = hand.HandwrittenDigitRecognizer()
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
            
            # Bind mouse events
            self.canvas.bind("<Button-1>", self.start_drawing)
            self.canvas.bind("<B1-Motion>", self.draw)
            self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)
            
            # Prediction callback
            self.predict_callback = None
        else:
            # Standalone mode
            self.window = tk.Tk()
            self.window.title("Draw a Digit")
            
            # Set window size and position
            self.window.geometry("320x400")
            self.window.resizable(False, False)
            
            # Create canvas
            self.canvas = tk.Canvas(
                self.window, 
                width=280, 
                height=280, 
                bg='white', 
                highlightthickness=1, 
                highlightbackground="black",
                cursor="pencil"
            )
            self.canvas.pack(pady=10)
            
            # Set canvas properties
            self.canvas.config(
                insertwidth=0,
                insertborderwidth=0,
                insertbackground='black'
            )
            
            # Create buttons frame
            button_frame = tk.Frame(self.window)
            button_frame.pack(pady=5)
            
            # Create buttons
            self.clear_button = tk.Button(button_frame, text="Clear", command=self.clear_canvas, width=10)
            self.clear_button.pack(side=tk.LEFT, padx=5)
            
            self.predict_button = tk.Button(button_frame, text="Predict", command=self.predict_digit, width=10)
            self.predict_button.pack(side=tk.LEFT, padx=5)
            
            # Initialize drawing state
            self.drawing = False
            self.last_x = None
            self.last_y = None
            self.points = []  # List to store points for smoothing
            
            # Create recognizer
            try:
                self.recognizer = hand.HandwrittenDigitRecognizer()
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
            
            # Create result label
            self.result_label = tk.Label(self.window, text="", font=("Arial", 14, "bold"))
            self.result_label.pack(pady=10)
            
            # Bind mouse events
            self.canvas.bind("<Button-1>", self.start_drawing)
            self.canvas.bind("<B1-Motion>", self.draw)
            self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)
            
            # Start application
            self.window.mainloop()

    def predict_digit(self):
        """Predict the drawn digit"""
        try:
            # Create a white image with the same size as canvas
            image = Image.new('L', (280, 280), color='white')
            draw = ImageDraw.Draw(image)
            
            # Get all items from canvas
            items = self.canvas.find_all()
            
            # Draw each item onto the image
            for item in items:
                coords = self.canvas.coords(item)
                if len(coords) == 4:  # Line
                    draw.line(coords, fill='black', width=20)
                elif len(coords) == 5:  # Oval
                    draw.ellipse(coords[:4], fill='black')
            
            # Resize to 28x28
            resized = image.resize((28, 28), Image.Resampling.LANCZOS)
            
            # Save for prediction
            current_dir = os.path.dirname(os.path.abspath(__file__))
            image_path = os.path.join(current_dir, "test_image.png")
            
            # Ensure directory exists
            os.makedirs(current_dir, exist_ok=True)
            
            # Save the image
            resized.save(image_path)
            
            # Make prediction
            digit, predictions = self.recognizer.predict(image_path)
            
            # Call callback if set
            if self.predict_callback:
                self.predict_callback(digit, predictions[0].split(': ')[1], predictions)
            
            # Clean up temporary files
            try:
                os.remove(image_path)
            except:
                pass
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            if self.predict_callback:
                self.predict_callback(None, "0%", ["Error: Failed to predict"])
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")

    def run(self):
        """Start the application"""
        if hasattr(self, 'window'):
            self.window.mainloop()

    def smooth_points(self, points, tension=0.5, num_points=100):
        """Smooth a list of points using cubic Bézier curves"""
        if len(points) < 2:
            return points
            
        def point(t):
            """Calculate point on Bézier curve"""
            x = 0
            y = 0
            n = len(points) - 1
            for i in range(len(points)):
                # Calculate Bézier coefficients
                b = (1 - t) ** (n - i) * t ** i * math.comb(n, i)
                x += b * points[i][0]
                y += b * points[i][1]
            return (x, y)
            
        # Generate points along the curve
        smoothed = []
        for i in range(num_points):
            t = i / (num_points - 1)
            smoothed.append(point(t))
            
        return smoothed

    def start_drawing(self, event):
        """Start drawing when mouse button is pressed"""
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
        
        # Draw initial dot with a nice size
        self.canvas.create_oval(
            self.last_x - 12, self.last_y - 12,
            self.last_x + 12, self.last_y + 12,
            fill='black', outline='black'
        )

    def draw(self, event):
        """Draw while mouse is moving"""
        if not self.drawing:
            return
            
        x, y = event.x, event.y
        
        # Draw line segment from last point to current point
        self.canvas.create_line(
            self.last_x, self.last_y, x, y,
            width=20, fill='black',
            capstyle=tk.ROUND, smooth=True,
            splinesteps=128
        )
        
        # Draw small dot at current position for visual feedback
        self.canvas.create_oval(
            x - 2, y - 2, x + 2, y + 2,
            fill='black', outline='black'
        )
        
        # Update last position
        self.last_x = x
        self.last_y = y

    def stop_drawing(self, event):
        """Stop drawing when mouse button is released"""
        self.drawing = False
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        """Clear the canvas"""
        self.canvas.delete("all")
        self.result_label.config(text="", fg='black')
        self.points = []  # Reset points list

    def predict_digit(self):
        """Predict the drawn digit"""
        try:
            # Create a white image with the same size as canvas
            image = Image.new('L', (280, 280), color='white')
            draw = ImageDraw.Draw(image)
            
            # Get all items from canvas
            items = self.canvas.find_all()
            
            # Draw each item onto the image
            for item in items:
                coords = self.canvas.coords(item)
                if len(coords) == 4:  # Line
                    draw.line(coords, fill='black', width=20)
                elif len(coords) == 5:  # Oval
                    draw.ellipse(coords[:4], fill='black')
            
            # Resize to 28x28
            resized = image.resize((28, 28), Image.Resampling.LANCZOS)
            
            # Save for prediction
            current_dir = os.path.dirname(os.path.abspath(__file__))
            image_path = os.path.join(current_dir, "test_image.png")
            
            # Ensure directory exists
            os.makedirs(current_dir, exist_ok=True)
            
            # Save the image
            resized.save(image_path)
            
            # Make prediction
            digit, predictions = self.recognizer.predict(image_path)
            
            # Format predictions
            predictions_text = "\n".join(predictions)
            self.result_label.config(text=f"Top predictions:\n{predictions_text}")
            
            # Change label color based on confidence
            if float(predictions[0].split(': ')[1].strip('%')) > 90:
                self.result_label.config(fg='green')
            else:
                self.result_label.config(fg='black')
                
            # Clean up temporary files
            try:
                os.remove(image_path)
            except:
                pass
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            self.result_label.config(text=f"Error: {str(e)}", fg='red')
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            if float(predictions[0].split(': ')[1].strip('%')) > 90:
                self.result_label.config(fg='green')
            else:
                self.result_label.config(fg='black')
                
            # Clean up temporary files
            os.remove("canvas.ps")
            os.remove("test_image.png")
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            self.result_label.config(text=f"Error: {str(e)}", fg='red')
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
        resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Save image for prediction
        resized.save("test_image.png")
        
        # Make prediction
        digit, predictions = self.recognizer.predict("test_image.png")
        
        # Format predictions
        predictions_text = "\n".join(predictions)
        self.result_label.config(text=f"Top predictions:\n{predictions_text}")
        
        # Change label color based on confidence
        if float(predictions[0].split(': ')[1].strip('%')) > 90:
            self.result_label.config(fg='green')
        else:
            self.result_label.config(fg='black')

    def run(self):
        """Start the application"""
        self.window.mainloop()

if __name__ == "__main__":
    try:
        drawer = DigitDrawer()
        drawer.run()
    except Exception as e:
        print(f"Application error: {str(e)}")
        messagebox.showerror("Error", f"Application failed to start: {str(e)}")
