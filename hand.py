import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import random

# Load the MNIST handwritten digit dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize the data (values between 0 and 1)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Build a simple neural network model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28, 1)),        # Flatten 28x28 images to 784
    keras.layers.Dense(128, activation='relu'),           # Hidden layer
    keras.layers.Dense(10, activation='softmax')          # Output layer for 10 digits
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Keep labels in sparse format
y_train = y_train.astype('int32')
y_test = y_test.astype('int32')

class HandwrittenDigitRecognizer:
    def __init__(self):
        self.model = model
        self.train_model()  # Train model on initialization

    def train_model(self):
        """Train the model if it hasn't been trained yet"""
        try:
            # Try to load existing model
            self.model = keras.models.load_model('handwritten_digit_recognizer.h5')
            print("Loaded existing model")
        except:
            # Train new model if it doesn't exist
            print("Training new model...")
            history = self.train(epochs=10)
            self.save_model()
            print("Model trained and saved")
            return history

    def train(self, epochs=10):
        history = self.model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
        y_test_sparse = np.argmax(y_test, axis=1)
        
        history = self.model.fit(
            x_train, y_train_sparse,
            epochs=epochs,
            validation_data=(x_test, y_test_sparse)
        )
        return history

    def train_model(self):
        """Train the model if it hasn't been trained yet"""
        try:
            # Try to load existing model
            self.model = keras.models.load_model('handwritten_digit_recognizer.h5')
            print("Loaded existing model")
        except:
            # Train new model if it doesn't exist
            print("Training new model...")
            history = self.train(epochs=10)
            self.save_model()
            print("Model trained and saved")
            return history

    def predict(self, image_path):
        """Predict digit from image path"""
        # Load and preprocess the image
        img = keras.preprocessing.image.load_img(image_path, color_mode='grayscale')
        img = keras.preprocessing.image.img_to_array(img)
        
        # Invert colors since our model was trained on black digits on white background
        img = 255 - img
        
        # Normalize to 0-1 range
        img = img.astype('float32') / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        # Make prediction
        prediction = self.model.predict(img)
        
        # Get top 3 predictions
        top3_indices = prediction[0].argsort()[-3:][::-1]
        top3_probs = prediction[0][top3_indices]
        
        # Format the results
        results = []
        for idx, prob in zip(top3_indices, top3_probs):
            results.append(f"{idx}: {prob * 100:.2f}%")
            
        # Clean up temporary file
        import os
        try:
            os.remove(image_path)
        except:
            pass
        
        # Return the top prediction and all results
        return top3_indices[0], results

    def evaluate(self):
        """Evaluate the model on test data"""
        # Convert labels back to sparse format for evaluation
        y_test_sparse = np.argmax(y_test, axis=1)
        test_loss, test_acc = self.model.evaluate(x_test, y_test_sparse)
        print(f"\nTest accuracy: {test_acc:.4f}")
        return test_acc

    def train_model(self):
        """Train the model if it hasn't been trained yet"""
        try:
            # Try to load existing model
            self.model = keras.models.load_model('handwritten_digit_recognizer.h5')
            print("Loaded existing model")
        except:
            # Train new model if it doesn't exist
            print("Training new model...")
            history = self.train(epochs=10)
            self.save_model()
            print("Model trained and saved")
            return history

    def predict(self, image_path):
        """Predict digit from image path"""
        # Load and preprocess the image
        img = keras.preprocessing.image.load_img(image_path, color_mode='grayscale')
        img = keras.preprocessing.image.img_to_array(img)
        
        # Invert colors since our model was trained on black digits on white background
        img = 255 - img
        
        # Normalize to 0-1 range
        img = img.astype('float32') / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        # Make prediction
        prediction = self.model.predict(img)
        
        # Get top 3 predictions
        top3_indices = prediction[0].argsort()[-3:][::-1]
        top3_probs = prediction[0][top3_indices]
        
        # Format the results
        results = []
        for idx, prob in zip(top3_indices, top3_probs):
            results.append(f"{idx}: {prob * 100:.2f}%")
            
        # Clean up temporary file
        import os
        try:
            os.remove(image_path)
        except:
            pass
        
        # Return the top prediction and all results
        return top3_indices[0], results

    def evaluate(self):
        """Evaluate the model on test data"""
        test_loss, test_acc = self.model.evaluate(x_test, y_test)
        print(f"\nTest accuracy: {test_acc:.4f}")
        return test_acc

    def train_model(self):
        """Train the model if it hasn't been trained yet"""
        try:
            # Try to load existing model
            self.model = keras.models.load_model('handwritten_digit_recognizer.h5')
            print("Loaded existing model")
        except:
            # Train new model if it doesn't exist
            print("Training new model...")
            history = self.train(epochs=10)
            self.save_model()
            print("Model trained and saved")
            return history

    def predict(self, image_path):
        """Predict digit from image path"""
        # Load and preprocess the image
        img = keras.preprocessing.image.load_img(image_path, color_mode='grayscale')
        img = keras.preprocessing.image.img_to_array(img)
        
        # Invert colors since our model was trained on black digits on white background
        img = 255 - img
        
        # Normalize to 0-1 range
        img = img.astype('float32') / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        # Make prediction
        prediction = self.model.predict(img)
        
        # Get top 3 predictions
        top3_indices = prediction[0].argsort()[-3:][::-1]
        top3_probs = prediction[0][top3_indices]
        
        # Format the results
        results = []
        for idx, prob in zip(top3_indices, top3_probs):
            results.append(f"{idx}: {prob * 100:.2f}%")
            
        # Clean up temporary file
        import os
        try:
            os.remove(image_path)
        except:
            pass
        
        # Return the top prediction and all results
        return top3_indices[0], results

    def evaluate(self):
        """Evaluate the model on test data"""

def train_model(self):
    """Train the model if it hasn't been trained yet"""
    try:
        # Try to load existing model
        self.model = keras.models.load_model('handwritten_digit_recognizer.h5')
        print("Loaded existing model")
    except:
        # Train new model if it doesn't exist
        print("Training new model...")
        history = self.train(epochs=10)
        self.save_model()
        print("Model trained and saved")
        return history

def predict(self, image_path):
    """Predict digit from image path"""
    # Load and preprocess the image
    img = keras.preprocessing.image.load_img(image_path, color_mode='grayscale')
    img = keras.preprocessing.image.img_to_array(img)
    
    # Invert colors since our model was trained on black digits on white background
    img = 255 - img
    
    # Normalize to 0-1 range
    img = img.astype('float32') / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    # Make prediction
    prediction = self.model.predict(img)
    
    # Get top 3 predictions
    top3_indices = prediction[0].argsort()[-3:][::-1]
    top3_probs = prediction[0][top3_indices]
    
    # Format the results
    results = []
    for idx, prob in zip(top3_indices, top3_probs):
        results.append(f"{idx}: {prob * 100:.2f}%")
        
    # Clean up temporary file
    import os
    try:
        os.remove(image_path)
    except:
        pass
        # Invert colors since our model was trained on black digits on white background
        img = 255 - img
        
        # Normalize to 0-1 range
        img = img / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        # Make prediction
        prediction = self.model.predict(img)
        
        # Get top 3 predictions
        top3_indices = prediction[0].argsort()[-3:][::-1]
        top3_probs = prediction[0][top3_indices]
        
        # Format the results
        results = []
        for idx, prob in zip(top3_indices, top3_probs):
            results.append(f"{idx}: {prob * 100:.2f}%")
            
        # Clean up temporary file
        import os
        try:
            os.remove(image_path)
        except:
            pass
        
        # Return the top prediction and all results
        return top3_indices[0], results

if __name__ == "__main__":
    # Create and train the model
    recognizer = HandwrittenDigitRecognizer()
    
    # Train and evaluate the model
    history = recognizer.train_model()
    
    # Plot training history
    if history:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    
    # Evaluate on test set
    recognizer.evaluate()
    
    # Test with a random digit from test set
    index = random.randint(0, len(x_test) - 1)
    test_image = x_test[index]
    
    # Add channel dimension for saving
    test_image_with_channel = np.expand_dims(test_image, axis=-1)
    
    # Save test image for prediction
    keras.preprocessing.image.save_img("test_image.png", test_image_with_channel)
    
    # Display the test image
    plt.imshow(test_image.reshape(28, 28), cmap='gray')
    plt.title(f"Actual digit: {y_test[index]}")
    plt.show()
    
    # Make prediction
    predicted_digit, predictions = recognizer.predict("test_image.png")
    print(f"\nPredicted digit: {predicted_digit}")
    print("Top predictions:")
    for pred in predictions:
        print(pred)
