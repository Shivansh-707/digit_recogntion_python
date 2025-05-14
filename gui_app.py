import sys
import numpy as np
import tensorflow as tf
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QPushButton, QLabel, QGraphicsView, QGraphicsScene, QGraphicsItem
)
from PIL import Image
from io import BytesIO

class DigitRecognizerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        try:
            self.model = tf.keras.models.load_model('digit_model.h5')
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

        self.setWindowTitle('Digit Recognizer')
        self.setGeometry(100, 100, 400, 450)
        self.init_ui()

    def init_ui(self):
        self.central_widget = QWidget(self) #container for other UI elements 
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget) #stacks element vertically 

        # Scene and view for drawing
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)
        self.view.setRenderHint(QPainter.Antialiasing) #render hints improves drawing quailty 
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.scene.setSceneRect(0, 0, 280, 280)

        self.canvas = Canvas() #custom class canvas (later in the code)
        self.scene.addItem(self.canvas)

        self.predict_button = QPushButton('Predict Digit', self)
        self.predict_button.clicked.connect(self.predict_digit) #when clicked it calls the predict_digit
        self.clear_button = QPushButton('Clear', self)
        self.clear_button.clicked.connect(self.clear_canvas) #when clicked it calls clear canvas 

        self.result_label = QLabel('Predicted Digit: None', self)
        self.result_label.setMinimumHeight(30) #to display the prediction result 
        self.result_label.setStyleSheet("font-size: 16px;") #originally 'predicted digit None'

        #put all the widgets on our window vertically 
        self.layout.addWidget(self.view)
        self.layout.addWidget(self.predict_button)
        self.layout.addWidget(self.clear_button)
        self.layout.addWidget(self.result_label)

    def predict_digit(self):
        try:
            pil_img = self.canvas.get_image() #get the image from the canvas
            img = pil_img.convert('L').resize((28, 28)) #resize to mnist input format
            # Invert colors: MNIST digits are white on black
            img = Image.eval(img, lambda x: 255 - x)
            # Threshold to clean image (black or white pixels only)
            img = img.point(lambda x: 0 if x < 128 else 255, '1')
            img = img.convert('L') # L means greyscale 

            # Save debug image to check what model sees
            img.save('debug_input.png')

            img_array = np.array(img).astype('float32') / 255.0 #convert image to numpy array 
            img_array = img_array.reshape(1, 28, 28, 1) #reshape to match model input 

            prediction = self.model.predict(img_array) #model makes the prediction (numpy array returned)
            predicted_digit = np.argmax(prediction) #gets digit with laargest value in the numpy array  
            confidence = np.max(prediction) * 100 #show confidence (confidence = np.max(prediction) * 100)

            print(f"Prediction vector: {prediction}")  # Debug print

            self.result_label.setText(f'Predicted Digit: {predicted_digit} (Confidence: {confidence:.2f}%)')

        except Exception as e:
            print(f"Prediction error: {e}")
            self.result_label.setText("Prediction failed - check terminal")

    def clear_canvas(self):
        self.canvas.clear()
        self.result_label.setText('Predicted Digit: None')

class Canvas(QGraphicsItem):
    def __init__(self):
        super().__init__()
        self.width_ = 280
        self.height_ = 280
        self.pixmap = QPixmap(self.width_, self.height_)
        self.pixmap.fill(Qt.white)
        self.drawing = False
        self.last_x, self.last_y = None, None

    def boundingRect(self):
        return QRectF(0, 0, self.width_, self.height_)

    def paint(self, painter, option, widget=None):
        painter.drawPixmap(0, 0, self.pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_x, self.last_y = event.pos().x(), event.pos().y()

    def mouseMoveEvent(self, event):
        if self.drawing:
            painter = QPainter(self.pixmap)
            pen = QPen(QColor(0, 0, 0), 25, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_x, self.last_y, event.pos().x(), event.pos().y())
            painter.end()
            self.last_x, self.last_y = event.pos().x(), event.pos().y()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def get_image(self):
        image = self.pixmap.toImage().convertToFormat(4)  # 4 = QImage.Format_RGBA8888
        width = image.width()
        height = image.height()
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.array(ptr).reshape((height, width, 4))
    
        # Convert to grayscale PIL Image
        gray = np.dot(arr[...,:3], [0.299, 0.587, 0.114])  # RGB to Grayscale
        pil_img = Image.fromarray(gray.astype(np.uint8))
        return pil_img


    def clear(self):
        self.pixmap.fill(Qt.white)
        self.update()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DigitRecognizerApp()
    window.show()
    sys.exit(app.exec_())
