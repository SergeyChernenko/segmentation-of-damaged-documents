import sys
from PyQt5.QtWidgets import QMainWindow, QPushButton, QWidget, QLabel, QLineEdit, QTextEdit, QGridLayout, QApplication, QFileDialog, QAction, QInputDialog, qApp, QHBoxLayout
from PyQt5.QtGui import QIcon, QPixmap
import Work as w
import Train_network as tn

class Example(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        w.ful_s()
        btn1 = QPushButton("Распознать", self)
        btn1.setGeometry(0, 0, 80, 80)
        btn1.move(630, 430)
        btn1.clicked.connect(self.buttonClicked)
        openFile = QAction(QIcon('open.png'), 'Выбрать файл', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Выбрать файл')
        openFile.triggered.connect(self.showDialog)
        exitButton = QAction(QIcon('exit24.png'), 'Выход', self)
        exitButton.setShortcut('Ctrl+P')
        exitButton.setStatusTip('Выход')
        exitButton.triggered.connect(self.close)
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&Опции')
        fileMenu.addAction(openFile)
        fileMenu.addAction(exitButton)
        self.le1 = QLineEdit(self)
        self.le1.setGeometry(0, 0, 250, 30)
        self.le1.move(400, 50)
        self.lbl = QLabel(self)
        self.lbl2 = QLabel(self)
        self.lbl.setGeometry(0, 0, 540, 420)
        self.lbl.move(750, 250)
        self.lbl2.setGeometry(0, 0, 540, 420)
        self.lbl2.move(50, 250)
        self.le = QTextEdit(self)
        self.le.setGeometry(300, 300, 500, 150)
        self.le.move(400, 80)
        self.showMaximized()
        self.setWindowTitle('OCR')
        self.show()

    def train_network(self):
        tn.train_con()
        tn.train_image_full()
    def showDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '/home')[0] 
        f = open(fname, 'r')
        print(f.name)
        self.le1.setText(str(f.name)) 
        self.pixmap2 = QPixmap(str(f.name))
        self.lbl2.setPixmap(self.pixmap2)
        w.im_scan=f.name
        
        
    def buttonClicked(self):
        w.train_con()
        w.train_image_full()
        self.pixmap = QPixmap("final.png")
        self.lbl.setPixmap(self.pixmap)
        w.recognition()
        file = open("testfile.txt", "r")
        self.le.setText(str(file.read())) 
            
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())