import sys
from PyQt5.Qt import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from recognize_doodle import *

train_images = np.load('yic_train_image.npy')
train_labels = np.load('yic_train_label.npy')
test_images = np.load('yic_test_image.npy')
test_labels = np.load('yic_test_label.npy')
train_images,test_images = train_images/255.0, test_images/255.0
model = getModel_CNN(train_images, train_labels, test_images, test_labels)

class PaintBoard(QWidget):
    def __init__(self, Parent=None):
        super().__init__(Parent)
        self.__InitData()
        self.__InitView()

    def __InitData(self):
        self.__size = QSize(140, 140)
        self.__board = QPixmap(self.__size)
        self.__board.fill(Qt.white)

        self.__IsEmpty = True
        self.EraserMode = False
        self.__lastPos = QPoint(0, 0)
        self.__currentPos = QPoint(0, 0)
        self.__painter = QPainter()
        self.__thickness = 4
        self.__penColor = QColor("black")
        self.__colorList = QColor.colorNames()

    def __InitView(self):
        self.setFixedSize(self.__size)

    def Clear(self):
        self.__board.fill(Qt.white)
        self.update()
        self.__IsEmpty = True

    def ChangePenColor(self, color="black"):
        self.__penColor = QColor(color)

    def ChangePenThickness(self, thickness=10):
        self.__thickness = thickness

    def IsEmpty(self):
        return self.__IsEmpty

    def GetContentAsQImage(self):
        image = self.__board.toImage()
        return image

    def paintEvent(self, paintEvent):
        self.__painter.begin(self)
        self.__painter.drawPixmap(0, 0, self.__board)
        self.__painter.end()

    def mousePressEvent(self, mouseEvent):
        self.__currentPos = mouseEvent.pos()
        self.__lastPos = self.__currentPos

    def mouseMoveEvent(self, mouseEvent):
        self.__currentPos = mouseEvent.pos()
        self.__painter.begin(self.__board)

        if self.EraserMode == False:
            self.__painter.setPen(QPen(self.__penColor, self.__thickness))
        else:
            self.__painter.setPen(QPen(Qt.white, 4))

        self.__painter.drawLine(self.__lastPos, self.__currentPos)
        self.__painter.end()
        self.__lastPos = self.__currentPos
        self.update()

    def mouseReleaseEvent(self, mouseEvent):
        self.__IsEmpty = False

class MainWidget(QWidget):

    def __init__(self, Parent=None):
        super().__init__(Parent)
        self.__InitData()
        self.__InitView()

    def __InitData(self):
        self.__paintBoard = PaintBoard(self)
        self.__colorList = QColor.colorNames()

    def __InitView(self):
        self.setFixedSize(400, 400)
        self.setWindowTitle("Recognize Doodle")
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(6)
        main_layout.addWidget(self.__paintBoard)
        sub_layout = QVBoxLayout()
        sub_layout.setContentsMargins(6, 6, 6, 6)

        self.__btn_Save = QPushButton("Compare")
        self.__btn_Save.setParent(self)
        self.__btn_Save.clicked.connect(self.on_btn_Save_Clicked)
        sub_layout.addWidget(self.__btn_Save)

        self.__btn_Clear = QPushButton("Clean")
        self.__btn_Clear.setParent(self)
        self.__btn_Clear.clicked.connect(self.clean_system)
        sub_layout.addWidget(self.__btn_Clear)

        self.__btn_Quit = QPushButton("Exit")
        self.__btn_Quit.setParent(self)
        self.__btn_Quit.clicked.connect(self.exit_system)
        sub_layout.addWidget(self.__btn_Quit)

        self.__cbtn_Eraser = QCheckBox("Earse")
        self.__cbtn_Eraser.setParent(self)
        self.__cbtn_Eraser.clicked.connect(self.on_cbtn_Eraser_clicked)
        sub_layout.addWidget(self.__cbtn_Eraser)

        splitter = QSplitter(self)
        sub_layout.addWidget(splitter)

        self.__label_penThickness = QLabel(self)
        self.__label_penThickness.setText("Brush Size")
        self.__label_penThickness.setFixedHeight(20)
        sub_layout.addWidget(self.__label_penThickness)

        self.__spinBox_penThickness = QSpinBox(self)
        self.__spinBox_penThickness.setMaximum(6)
        self.__spinBox_penThickness.setMinimum(0.2)
        self.__spinBox_penThickness.setValue(1)
        self.__spinBox_penThickness.setSingleStep(1)
        self.__spinBox_penThickness.valueChanged.connect(
            self.on_PenThicknessChange)
        sub_layout.addWidget(self.__spinBox_penThickness)

        self.__label_penColor = QLabel(self)
        self.__label_penColor.setText("Brush Color")
        self.__label_penColor.setFixedHeight(20)
        sub_layout.addWidget(self.__label_penColor)

        self.__btn_Color = QPushButton("PickColor")
        self.__btn_Color.setParent(self)
        self.__btn_Color.clicked.connect(self.color_picker)
        sub_layout.addWidget(self.__btn_Color)

        main_layout.addLayout(sub_layout)

    def exit_system(self):
        sys.exit()

    def clean_system(self):
        self.__paintBoard.Clear()
        cv2.destroyAllWindows()

    def color_picker(self):
        self.__paintBoard.ChangePenColor(QColorDialog.getColor())

    def on_PenThicknessChange(self):
        penThickness = self.__spinBox_penThickness.value()
        self.__paintBoard.ChangePenThickness(penThickness)

    def on_btn_Save_Clicked(self):
        image = self.__paintBoard.GetContentAsQImage()
        image.save('image.png', "PNG")
        img = cv2.imread('image.png')
        compareImages(img, model)

    def on_cbtn_Eraser_clicked(self):
        if self.__cbtn_Eraser.isChecked():
            self.__paintBoard.EraserMode = True
        else:
            self.__paintBoard.EraserMode = False

    def Quit(self):
        self.close()


def main():
    app = QApplication(sys.argv)
    mainWidget = MainWidget()
    mainWidget.show()
    exit(app.exec_())


if __name__ == '__main__':
    main()
