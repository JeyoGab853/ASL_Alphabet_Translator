import sys
from pathlib import Path
import os

import PyQt5.QtCore
import PyQt5.QtGui
import cv2
import numpy as np
import pyautogui
import keyboard
import time

from translatorModule import Translator

from PyQt5.QtCore import Qt, QThread, pyqtSignal,QObject, pyqtSlot, QTimer
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTextEdit, QComboBox,QMessageBox,
                             QLabel, QWidget, QBoxLayout, QGridLayout, QVBoxLayout, QHBoxLayout, QPushButton, QFrame)
from PyQt5.QtGui import QLinearGradient
from PyQt5.QtGui import QFont, QPixmap, QImage

def get_resource_path(relative_path):
    try:
        base_path = Path(sys._MEIPASS)
    except AttributeError:
        base_path = Path(__file__).resolve().parent

    return base_path / relative_path

def numpytoqimage(array: np.ndarray) -> QImage:
    array = np.ascontiguousarray(array, dtype=np.uint8)
    if len(array.shape) == 2:
        height, width = array.shape
        bytesPerLine = width
        return QImage(
            array.data,
            width,
            height,
            bytesPerLine,
            QImage.Format_Grayscale8
        ).copy()

    elif len(array.shape) == 3:
        height, width, channels = array.shape
        bytesPerLine = channels * width

        if channels == 3:
            return QImage(
                array.data,
                width,
                height,
                bytesPerLine,
                QImage.Format_RGB888
            ).copy()

        elif channels == 4:
            return QImage(
                array.data,
                width,
                height,
                bytesPerLine,
                QImage.Format_ARGB32
            ).copy()

    return QImage()

class Worker(QObject):
    imageFrameSignal = pyqtSignal(QPixmap)
    sendLetterSignal = pyqtSignal(str)
    sendCropSignal = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self.Translator = Translator()

        self.cropping = False
        self._running = True
        self.startCrop = False
        self.translating = False
        self.currentDims = []

    @pyqtSlot()
    def showImage(self):

        x1,y1,w,h = tuple(self.currentDims)

        img = pyautogui.screenshot()
        croppedImg = np.array(img)[y1: h, x1: w]

        self.Translator.frame = croppedImg

        finalImage = numpytoqimage(croppedImg)
        newPixMap = QPixmap(finalImage)

        self.imageFrameSignal.emit(newPixMap)

    @pyqtSlot()
    def mainthread(self):
        self.timer = QTimer(self)

        self.timer.timeout.connect(self.looping)
        self.timer.start(8)

    def looping(self):
        # print(self.startCrop)
        if not self._running:
            self.timer.stop()
            return

        if self.translating:
            self.sendLetterSignal.emit(self.Translator.translate())

        if len(self.currentDims) >= 4:
            self.showImage()

        if self.cropping:
            self.crop()

    def crop(self):
        if keyboard.is_pressed('r'):
            self.currentDims.extend(pyautogui.position())
            print(pyautogui.position())

            if len(self.currentDims) == 2:
                self.sendCropSignal.emit('press R to Crop (2)')
            elif len(self.currentDims) == 4:
                self.sendCropSignal.emit('Crop')

            time.sleep(.2)

        if len(self.currentDims) >= 4:
            self.cropping = False

    @pyqtSlot()
    def stop(self):
        print('stopped')
        self.timer.stop()
        self.cropping = False
        self._running = False
        self.translating = False

    @pyqtSlot()
    def enablecrop(self):
        self.cropping = True
        self.currentDims.clear()

    @pyqtSlot(bool)
    def toggletranslation(self,boolvalue):
        print('Toggled:',boolvalue)
        self.Translator.dataSequence.clear()
        self.Translator.currentLetter = None
        self.translating = boolvalue


class MainWindow(QMainWindow):
    videoSignal = pyqtSignal(int)
    sendCropSignal = pyqtSignal()
    translateSignal = pyqtSignal(bool)
    stopSignal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Language Translator")

        self.initUI()
        self.OptionTranslation()
        self.style()

        self.setupThread()
        self.HideButton2.clicked.connect(self.hideButton2Pressed)
        self.CropButton.clicked.connect(self.startCrop)
        self.ButtonTranslate.clicked.connect(self.starttranslate)
        self.ClearButton.clicked.connect(self.clearOutput)

    def starttranslate(self):
        if self.ButtonTranslate.text() == 'Translate':
            self.ButtonTranslate.setText('Stop')
            self.translateSignal.emit(True)
        else:
            self.ButtonTranslate.setText('Translate')
            self.translateSignal.emit(False)

    def startCrop(self):
        self.CropButton.setEnabled(False)
        self.CropButton.setText('press R to Crop (1)')
        self.sendCropSignal.emit()

    def setupThread(self):
        self.thread = QThread()
        self.worker = Worker()

        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.mainthread)
        self.sendCropSignal.connect(self.worker.enablecrop)
        self.translateSignal.connect(self.worker.toggletranslation)
        self.stopSignal.connect(self.worker.stop)

        self.worker.imageFrameSignal.connect(self.recieveImageData)
        self.worker.sendLetterSignal.connect(self.recieveLetterData)
        self.worker.sendCropSignal.connect(self.recieveCropSignal)

        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def clearOutput(self):
        self.OutputBox.clear()

    def closeEvent(self, event):
        self.stopSignal.emit()

        if self.thread.isRunning():
            self.worker.stop()

    def recieveLetterData(self,value: str):
        inputBoxText = self.OutputBox.toPlainText()

        if not value:
            return

        value = value.upper()

        if inputBoxText:
            self.OutputBox.setText(inputBoxText + value)
        else:
            self.OutputBox.setText(value)

    def recieveImageData(self,pixmap):
        largerDim = max(pixmap.width(), pixmap.height())
        differenceToMin = largerDim - min(largerDim, 250)

        scaledHeight = int(round(abs(pixmap.height() - differenceToMin)))
        scaledWidth = int(round(abs(pixmap.width() - differenceToMin)))

        scaledPixmap = pixmap.scaled(scaledWidth, scaledHeight, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.ImageBox.setPixmap(scaledPixmap)

    def recieveCropSignal(self,stringValue):
        self.CropButton.setText(stringValue)

        if stringValue == 'Crop':
            self.CropButton.setEnabled(True)

    def initUI(self):
        central_widget = QWidget()

        self.setCentralWidget(central_widget)

        self.InputBox = QTextEdit()
        self.ImageBox = QLabel(self)
        self.OutputBox = QTextEdit()

        self.InputBox.hide()

        self.TranslateOption = QComboBox()
        self.TranslateOption.addItem("ASL to Speech")
        self.TranslateOption.addItem("Speech to ASL")

        self.TranslateOption.hide()

        self.ButtonTranslate = QPushButton("Translate")
        self.CropButton = QPushButton('Crop')

        self.CropButton.setFixedSize(100,30)
        self.ButtonTranslate.setFixedSize(250,50)

        self.ClearButton = QPushButton('Clear Output')
        self.ClearButton.setFixedSize(150,25)

        self.TitleLabel = QLabel("Sign Language Translator")
        self.LabelOutput = QLabel("Output")
        self.LabelInput = QLabel("Input")

        #INPUTCONTAINER-----------------------------
        self.InputContainer  = QWidget()
        LayoutInput = QVBoxLayout()

        LayoutInput.addWidget(self.LabelInput)
        LayoutInput.addWidget(self.InputBox)
        LayoutInput.addWidget(self.ImageBox)

        self.InputContainer.setLayout(LayoutInput)
        #-----------------------------


        self.HideButton2 = QPushButton("-")

        self.HideButton2.setFixedSize(20,20)

        pixmap = QPixmap(os.path.join(get_resource_path('APPimages'),'PlaceHolderIMG.png'))

        largerDim = max(pixmap.width(),pixmap.height())
        differenceToMin = largerDim - min(largerDim,300)

        scaledHeight = int(round(abs(pixmap.height() - differenceToMin)))
        scaledWidth = int(round(abs(pixmap.width() - differenceToMin)))

        scaled_pixmap = pixmap.scaled(scaledWidth, scaledHeight, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.ImageBox.setPixmap(scaled_pixmap)
        self.ImageBox.setAlignment(Qt.AlignCenter)

        self.master = QVBoxLayout()

        col1 = QVBoxLayout()
        col2 = QVBoxLayout()

        self.Container1 = QFrame()
        self.Container2 = QFrame()

        col1.addWidget(self.TitleLabel)
        col1.addWidget(self.TranslateOption)
        # col1.addWidget(self.ButtonTranslate)

        col2.addWidget(self.HideButton2)
        col2.addWidget(self.InputContainer)
        col2.addWidget(self.LabelOutput)
        col2.addWidget(self.ClearButton,alignment=Qt.AlignmentFlag.AlignHCenter)
        col2.addWidget(self.OutputBox)

        self.Container1.setLayout(col1)
        self.Container2.setLayout(col2)

        self.master.addWidget(self.Container1,10,alignment=Qt.AlignmentFlag.AlignHCenter)
        self.master.addWidget(self.ButtonTranslate,20,alignment=Qt.AlignmentFlag.AlignHCenter)
        self.master.addWidget(self.CropButton,alignment=Qt.AlignmentFlag.AlignHCenter)
        self.master.addWidget(self.Container2,70)

        central_widget.setLayout(self.master)

    def style(self):
        self.TitleLabel.setStyleSheet("""font-size: 20px;
                                        padding: 5px;""")

        self.LabelOutput.setStyleSheet("font-size: 15px; font-weight: bold; background-color: #ecd9ff; padding:0 5px;")
        self.LabelInput.setStyleSheet("font-size: 15px; font-weight: bold; background-color: #ecd9ff; padding:0 5px;")


        self.Container1.setObjectName("Container")
        self.Container2.setObjectName("Container")

        self.Container1.setFixedSize(280,100)

        self.ButtonTranslate.setStyleSheet("""
                            QPushButton {
                                background-color: #7e5c94;
                                color: #e8e8e8;
                                font-weight: bold;
                                font-size: 15px;
                                border-radius: 5px;
                                border: 2px solid #5d436e;
                                padding: 5px;
                            }
                            QPushButton:hover {
                                background-color: #5d436e;
                                color: grey
                            }
                            QPushButton:pressed {
                                background-color: #3c2c47;

                            }
                        """)

        self.setStyleSheet("""
            #Container {
                border: 1px outset grey;
                border-radius: 5px;
            }
            """)

    def hideButton2Pressed(self):
        Button2 = self.HideButton2
        InputContainer = self.InputContainer

        if Button2.text() == "-":
            Button2.setText("+")
            InputContainer.hide()

        elif Button2.text() == "+":
            Button2.setText("-")
            InputContainer.show()

    #----------------------------------------------Unused func
    def OptionTranslation(self):
        self.TranslateOption.currentTextChanged.connect(self.ontextchanged)
    #-------

    def ontextchanged(self,text=""):
        if text == "Speech to ASL":
            self.InputBox.show()
            self.ImageBox.hide()
            self.CropButton.hide()
        else:
            self.InputBox.hide()
            self.ImageBox.show()
            self.CropButton.show()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
