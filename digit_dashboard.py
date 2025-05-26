import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageDraw
import numpy as np
import os
import hand
import io
from draw_digit import DigitDrawer

class DigitDashboard:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Handwritten Digit Recognition Dashboard")
        self.window.geometry("800x600")
        self.window.configure(bg='#f0f0f0')
        
        # Create main frame
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create header
        header_frame = ttk.Frame(main_frame)
        header_frame.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        header_label = ttk.Label(
            header_frame,
            text="Handwritten Digit Recognition",
            font=("Arial", 24, "bold")
        )
        header_label.pack()
        
        # Create drawing area
        self.drawing_area = ttk.LabelFrame(
            main_frame,
            text="Draw a Digit",
            padding="10"
        )
        self.drawing_area.grid(row=1, column=0, padx=10, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Initialize digit drawer
        self.drawer = DigitDrawer(self.drawing_area)
        self.drawer.canvas.pack()
        
        # Create prediction area
        prediction_frame = ttk.LabelFrame(
            main_frame,
            text="Prediction Results",
            padding="10"
        )
        prediction_frame.grid(row=1, column=1, padx=10, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Prediction display
        self.prediction_label = ttk.Label(
            prediction_frame,
            text="",
            font=("Arial", 14, "bold"),
            anchor=tk.CENTER,
            padding=(0, 20)
        )
        self.prediction_label.pack(fill=tk.X)
        
        # Confidence display
        self.confidence_label = ttk.Label(
            prediction_frame,
            text="",
            font=("Arial", 12),
            anchor=tk.CENTER,
            padding=(0, 10)
        )
        self.confidence_label.pack(fill=tk.X)
        
        # Statistics display
        stats_frame = ttk.LabelFrame(
            prediction_frame,
            text="Statistics",
            padding="5"
        )
        stats_frame.pack(fill=tk.X, pady=(20, 0))
        
        self.stats_labels = []
        for i in range(3):
            label = ttk.Label(
                stats_frame,
                text="",
                font=("Arial", 10)
            )
            label.pack(fill=tk.X, pady=5)
            self.stats_labels.append(label)
        
        # Create control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=(20, 0))
        
        # Clear button
        clear_button = ttk.Button(
            button_frame,
            text="Clear Canvas",
            command=self.drawer.clear_canvas,
            width=15
        )
        clear_button.pack(side=tk.LEFT, padx=5)
        
        # Predict button
        predict_button = ttk.Button(
            button_frame,
            text="Predict Digit",
            command=self.predict,
            width=15
        )
        predict_button.pack(side=tk.LEFT, padx=5)
        
        # Exit button
        exit_button = ttk.Button(
            button_frame,
            text="Exit",
            command=self.window.quit,
            width=15
        )
        exit_button.pack(side=tk.LEFT, padx=5)
        
        # Configure grid weights
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Bind prediction callback
        self.drawer.predict_callback = self.update_prediction
        
    def predict(self):
        """Handle prediction request"""
        try:
            # Get canvas content
            items = self.drawer.canvas.find_all()
            if not items:
                messagebox.showerror("Error", "Please draw a digit first")
                return
                
            # Create a white image with the same size as canvas
            image = Image.new('L', (280, 280), color='white')
            draw = ImageDraw.Draw(image)
            
            # Draw each item onto the image
            for item in items:
                coords = self.drawer.canvas.coords(item)
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
            digit, predictions = self.drawer.recognizer.predict(image_path)
            
            # Update prediction display
            self.update_prediction(digit, predictions[0].split(': ')[1], predictions)
            
            # Clean up temporary files
            try:
                os.remove(image_path)
            except:
                pass
                
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            self.update_prediction(None, "0%", ["Error: Failed to predict"])
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def update_prediction(self, prediction, confidence, stats):
        """Update prediction display with results"""
        self.prediction_label.config(text=f"Predicted Digit: {prediction}")
        
        # Update confidence color based on value
        if float(confidence.strip('%')) > 90:
            self.confidence_label.config(
                text=f"Confidence: {confidence}",
                foreground="green"
            )
        else:
            self.confidence_label.config(
                text=f"Confidence: {confidence}",
                foreground="black"
            )
        
        # Update statistics
        for i, stat in enumerate(stats):
            self.stats_labels[i].config(text=stat)
            
    def run(self):
        """Start the application"""
        self.window.mainloop()
    
    def update_prediction(self, prediction, confidence, stats):
        """Update prediction display with results"""
        self.prediction_label.config(text=f"Predicted Digit: {prediction}")
        
        # Update confidence color based on value
        if float(confidence.strip('%')) > 90:
            self.confidence_label.config(
                text=f"Confidence: {confidence}",
                foreground="green"
            )
        else:
            self.confidence_label.config(
                text=f"Confidence: {confidence}",
                foreground="black"
            )
        
        # Update statistics
        for i, stat in enumerate(stats):
            self.stats_labels[i].config(text=stat)
            
    def run(self):
        """Start the application"""
        self.window.mainloop()

if __name__ == "__main__":
    try:
        dashboard = DigitDashboard()
        dashboard.drawer.predict_callback = dashboard.update_prediction
        dashboard.run()
    except Exception as e:
        print(f"Application error: {str(e)}")
        messagebox.showerror("Error", f"Application failed to start: {str(e)}")
