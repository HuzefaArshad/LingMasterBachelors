# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Spell_Grammer_Check.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
from autocorrect import Speller
from gingerit.gingerit import GingerIt


class Ui_MainWindow1(object):
    def SpellChecker(self):
            
          spell = Speller()
          spellfree=spell(self.textEdit.toPlainText().strip())
          
          if spellfree!=self.textEdit.toPlainText().strip():
                self.textEdit_3.append(spellfree)
                msg = QMessageBox()
                msg.setWindowTitle("Spelling Converter")
                msg.setText("Succesfully Checked All Spelling Mistakes!")
                msg.setIcon(QMessageBox.Information)
                x = msg.exec_()
          else:
                msg = QMessageBox()
                msg.setWindowTitle("Spelling Converter")
                msg.setText("Error in spelling Checker!")
                msg.setIcon(QMessageBox.Information)
                x = msg.exec_()




    def grammarcheck(self):
         text=self.textEdit_3.toPlainText().strip()
         parser = GingerIt()
         ai  =parser.parse(text)['result']       
         self.textEdit_4.append(ai)
         if ai!=text:
                msg = QMessageBox()
                msg.setWindowTitle("Grammar Correction")
                msg.setText("Succesfully Checked All Gramattical  Mistakes!")
                msg.setIcon(QMessageBox.Information)
                x = msg.exec_()
         else:
                msg = QMessageBox()
                msg.setWindowTitle("Grammar Correction")
                msg.setText("Error in Grammar Checker!")
                msg.setIcon(QMessageBox.Information)
                x = msg.exec_()
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(843, 445)
        MainWindow.setStyleSheet("background-color: #272743")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(30, 290, 81, 31))
        self.label_3.setStyleSheet("font-size:15pt;\n"
"color:rgb(255, 255, 253);\n"
"font: SMILEN;")
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(10, 320, 151, 31))
        self.label_4.setStyleSheet("font-size:15pt;\n"
"color:rgb(255, 255, 253);\n"
"font: SMILEN;")
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(20, 350, 111, 31))
        self.label_5.setStyleSheet("font-size:15pt;\n"
"color:rgb(255, 255, 253);\n"
"font: SMILEN;")
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(30, 380, 81, 31))
        self.label_6.setStyleSheet("font-size:15pt;\n"
"color:rgb(255, 255, 253);\n"
"font: SMILEN;")
        self.label_6.setObjectName("label_6")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(190, 10, 191, 31))
        self.label.setStyleSheet("font-size:15pt;\n"
"color:rgb(255, 255, 253);\n"
"font: SMILEN;")
        self.label.setObjectName("label")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(190, 40, 531, 111))
        self.textEdit.setStyleSheet("background-color:white;")
        self.textEdit.setObjectName("textEdit")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(730, 210, 111, 41))
        self.pushButton.setStyleSheet("background-color: rgb(186, 232, 232);\n"
"color: rgb(45, 102, 142);\n"
"font: SMILEN;\n"
"border-style:outset;\n"
"border-width:2px;\n"
"border-radius:10px;\n"
"font:bold 12px;\n"
"border-color:black;\n"
"padding:10px;")
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.SpellChecker)
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(190, 160, 301, 31))
        self.label_7.setStyleSheet("font-size:15pt;\n"
"color:rgb(255, 255, 253);\n"
"font: SMILEN;")
        self.label_7.setObjectName("label_7")
        self.textEdit_3 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_3.setGeometry(QtCore.QRect(190, 190, 531, 81))
        self.textEdit_3.setStyleSheet("background-color:white;")
        self.textEdit_3.setPlaceholderText("")
        self.textEdit_3.setObjectName("textEdit_3")
        self.textEdit_3.setReadOnly(True)
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(190, 280, 191, 31))
        self.label_8.setStyleSheet("font-size:15pt;\n"
"color:rgb(255, 255, 253);\n"
"font: SMILEN;")
        self.label_8.setObjectName("label_8")
        self.textEdit_4 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_4.setGeometry(QtCore.QRect(190, 310, 531, 121))
        self.textEdit_4.setStyleSheet("background-color:white;")
        self.textEdit_4.setPlaceholderText("")
        self.textEdit_4.setObjectName("textEdit_4")
        self.textEdit_4.setReadOnly(True)
        self.seperator_summary = QtWidgets.QFrame(self.centralwidget)
        self.seperator_summary.setGeometry(QtCore.QRect(170, 0, 21, 441))
        self.seperator_summary.setStyleSheet("width:70px")
        self.seperator_summary.setFrameShape(QtWidgets.QFrame.VLine)
        self.seperator_summary.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.seperator_summary.setObjectName("seperator_summary")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(730, 80, 111, 41))
        self.pushButton_2.setStyleSheet("background-color: rgb(186, 232, 232);\n"
"color: rgb(45, 102, 142);\n"
"font: SMILEN;\n"
"border-style:outset;\n"
"border-width:2px;\n"
"border-radius:10px;\n"
"font:bold 12px;\n"
"border-color:black;\n"
"padding:10px;")
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(730, 350, 111, 41))
        self.pushButton_3.setStyleSheet("background-color: rgb(186, 232, 232);\n"
"color: rgb(45, 102, 142);\n"
"font: SMILEN;\n"
"border-style:outset;\n"
"border-width:2px;\n"
"border-radius:10px;\n"
"font:bold 12px;\n"
"border-color:black;\n"
"padding:10px;")
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.clicked.connect(self.grammarcheck)
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(0, 20, 151, 121))
        self.label_9.setStyleSheet("background-color: #272743")
        self.label_9.setText("")
        self.label_9.setPixmap(QtGui.QPixmap("LM.png"))
        self.label_9.setScaledContents(True)
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(10, 170, 131, 101))
        self.label_10.setStyleSheet("background-color: #272743")
        self.label_10.setText("")
        self.label_10.setPixmap(QtGui.QPixmap("Check grammar and spelling error.png"))
        self.label_10.setScaledContents(True)
        self.label_10.setObjectName("label_10")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_3.setText(_translate("MainWindow", "CHECK"))
        self.label_4.setText(_translate("MainWindow", "GRAMMAR & "))
        self.label_5.setText(_translate("MainWindow", "SPELLING "))
        self.label_6.setText(_translate("MainWindow", "ERROR"))
        self.label.setText(_translate("MainWindow", "Enter Text Here:"))
        self.textEdit.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:7.8pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:8.25pt;\"><br /></p></body></html>"))
        self.textEdit.setPlaceholderText(_translate("MainWindow", "Write your sentence here.."))
        self.pushButton.setText(_translate("MainWindow", "Spell Check"))
        self.label_7.setText(_translate("MainWindow", "After Spelling Correction:"))
        self.textEdit_3.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:7.8pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:8.25pt;\"><br /></p></body></html>"))
        self.label_8.setText(_translate("MainWindow", "Error Free Text:"))
        self.textEdit_4.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:7.8pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:8.25pt;\"><br /></p></body></html>"))
        self.pushButton_2.setText(_translate("MainWindow", "Go To HOME"))
        self.pushButton_3.setText(_translate("MainWindow", "Grammar"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow1()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
