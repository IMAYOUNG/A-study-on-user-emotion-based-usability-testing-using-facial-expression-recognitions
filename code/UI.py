import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from FER import detect_emotion
from timeseriesOutput import plot
import pandas as pd

class Video(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Facial Expression Recognition")
        self.setFixedSize(420, 130)
        self.setStyleSheet("background-color: navy;")

        titleLabel = QLabel("SITLab FER", self)
        titleLabel.setAlignment(Qt.AlignCenter)
        titleLabel.setFont(QFont('Comic Sans Ms', 30))
        titleLabel.setStyleSheet("color: white;")

        self.videoButton = QPushButton('Select File path', self)
        self.videoButton.setFont(QFont('Consolas', 20))
        self.videoButton.setStyleSheet("QPushButton {color: white; background-color: rgb(58, 134, 255); border-radius: 5px;}"
                                        "QPushButton:pressed { background-color: rgb(30, 106, 255); }")
        self.videoButton.clicked.connect(self.toggleVideo)
        
        self.outputButton = QPushButton('Output', self)
        self.outputButton.setFont(QFont('Consolas', 20))
        self.outputButton.setStyleSheet("QPushButton {color: white; background-color: rgb(58, 134, 255); border-radius: 5px;}"
                                        "QPushButton:pressed { background-color: rgb(30, 106, 255); }")
        self.outputButton.clicked.connect(self.OutputGraph)

        quitButton = QPushButton('Exit', self)
        quitButton.setFont(QFont('Consolas', 20))
        quitButton.setStyleSheet("QPushButton {color: white; background-color: rgb(58, 134, 255); border-radius: 5px;}"
                                  "QPushButton:pressed { background-color: rgb(30, 106, 255); }")
        quitButton.clicked.connect(self.quitFunction)

        self.buttonLayout = QHBoxLayout()
        self.buttonLayout.addWidget(self.videoButton)
        self.buttonLayout.addWidget(self.outputButton)
        self.buttonLayout.addWidget(quitButton)

        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)
        
        layout = QVBoxLayout(centralWidget)
        layout.addWidget(titleLabel)
        layout.addLayout(self.buttonLayout)

        self.cap = None
        self.video_active = False
        self.output_window = None
        self.output_path = None

    def toggleVideo(self):
        if not self.video_active:
            self.selectOutputPath()
        else:
            self.video_active = False
            # self.videoButton.setText('FER On')

    def quitFunction(self):
        if self.video_active:
            self.video_active = False
        self.close()

    def OutputGraph(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Output CSV File", "", "CSV Files (*.csv)")
        if filename:
            self.output_path = filename
            try:
                plot(self.output_path)
            except pd.errors.ParserError:
                error_message = "Error parsing CSV file."
                msg_box = QMessageBox()
                msg_box.setWindowTitle("Error")
                msg_box.setText(error_message)
                msg_box.exec_()
    
    def selectOutputPath(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save CSV File", "", "CSV Files (*.csv)")
        if filename:
            self.output_path = filename
            self.video_active = True
            self.videoButton.setText('FER Off')
            
            detect_emotion(self, self.output_path)
            self.videoButton.setText('Select File path')  

    def closeEvent(self, event):
        if self.video_active:
            self.video_active = False
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Video()
    win.show()
    sys.exit(app.exec_())
