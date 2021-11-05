# -*- coding: utf-8 -*-

# импорт модулей pytorch
import torch
import torchvision.transforms as transform
from torchvision.models import resnet152
import torch.nn.functional as F

# импорт сторонних модулей
from PIL import Image
from imagenet_classes import *
import numpy as np
import sys

# импорт модулей PyQt
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore


class ClassifyThread(QtCore.QThread):
    '''
    Класс пользовательского потока для выполнения процесса классификации изображения
    '''
    # объект пользовательского сигнала
    classifySignal = QtCore.pyqtSignal(dict)

    def __init__(self, path_to_image: str, parent=None):
        '''
        Конструктор класса пользовательского потока

        Параметры: 
            path_to_image (str): путь к изображению, которого необходимо классифицировать

        '''
        QtCore.QThread.__init__(self, parent)
        self.path_to_image = path_to_image


    def run(self):
        '''
        Метод, который выполняется при запуске потока
        В нем непосредственно выполняется классификация 
        '''
        # создаем объект трансформации изображения
        transforms = transform.Compose([
                transform.Resize((224, 224)),
                transform.ToTensor(),
                transform.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        # открываем изображение через PIL.Image
        opened_image      = Image.open(self.path_to_image)
        # применяем трансформации к изображению
        transformed_image = transforms(opened_image)
        # добавляем размерность батча к изображению, равную 1 
        unsqueezed_image  = transformed_image.unsqueeze_(0)
        # по умолчанию устройством будет CPU
        device            = "cpu"
        # инициализируем предобученный классификатор ResNet152 и переводим его в режим валидации
        resnet            = resnet152(pretrained=True)
        resnet.eval()
        # перемещаем изображение на устройство
        unsqueezed_image  = unsqueezed_image.to(device)
        # формируем тензор предсказаний и прогоняем его через softmax
        predictions       = F.softmax(resnet(unsqueezed_image), dim=1)
        # находим индекс наибольшего элемента тензора, означающего наибольшее значение вероятности
        prediction        = int(predictions.argmax())
        # находим название класса из словаря классов ImageNet по значению prediction
        # если названия классов по ключу prediction перечислены через запятую. выбираем самый первый
        predicted_label   = imagenet_classes[prediction].split(",")[0]
        # находим значение вероятности по prediction
        confidence        = float(predictions[0][prediction])

        # передаем данные о классифицированном изображении наружу через пользовательский сигнал
        self.classifySignal.emit({
                    "ImageNet class id": prediction,
                     "Predicted class": predicted_label,
                     "Confidence": confidence}
                )


class ClassifierApp(QMainWindow):
    '''
    Класс приложения ClassifierApp
    '''
    def __init__(self):
        '''
        Конструктор класса ClassifierApp
        У объекта этого класса вызывается метод initUI,
        который непосредственно инициализирует интерфейс
         '''
        super(ClassifierApp, self).__init__()
        self.initUI()
    
    
    def initUI(self):
        '''
        Метод инициализации интерфейса
        '''
        # загружаем интерфейс из UI-файла
        uic.loadUi("qt_image_classifier_ui.ui", self)
        # прикрепляем к событию нажатий кнопок методы
        self.openImage.clicked.connect(self.chooseImage) # по нажатию на кнопку выведется openfiledialog с просьбой выбрать изображение
        self.clear.clicked.connect(self.clear_fields) # по нажатию на эту кнопку очистятся все поля
        self.processImage.clicked.connect(self.classify) # по нажатию на эту кнопку будет запущен классификатор на загруженном изображении
        self.processImage.setEnabled(False) # блокируем эту кнопку, т. к. классифицировать пока нечего
        self.imageLabel.setStyleSheet("border: 1px solid black") # визуализируем границы imageLabel
        self.menuAbout_Qt.addAction("About Qt") # добавляем к кнопке панели инструментов About элемент AboutQt
        self.menuAbout_Qt.triggered.connect(self.aboutQt) # по нажатию на которого будет выводиться справка о Qt
        
        
    def chooseImage(self):
        '''
        Метод для выбора изображения
        '''
        # выводим openfiledialog и получаем путь к изображению
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(
                self,
                "QFileDialog.getOpenFileName()",
                "",
                "All Files (*);;PNG files (*.png);;JPEG files (*.jpg)",
                options=options
            )
        # помещаем путь к изображению в текстовое поле
        self.textfield.setText(filename)
        # зсгружаем изображение по пути
        self.load_image(filename)
        # активируем кнопку для классификации
        self.processImage.setEnabled(True)


    def clear_fields(self):
        '''
        Метод для очистки всех лейблов и полей
        '''
        self.textfield.clear()
        self.imageLabel.clear()
        self.imageNetId.setText("ImageNet id: ")
        self.predictedClass.setText("Predicted class: ")
        self.confidence_label.setText("Confidence: ")
        # блокируем кнопку для классификации
        self.processImage.setEnabled(False)
        
        
    def classify(self):
        '''
        Метод, вызывающий поток для классификации изображения
        '''
        path_to_image       = self.textfield.toPlainText()
        self.classifyThread = ClassifyThread(path_to_image)
        self.classifyThread.classifySignal.connect(self.finish_thread, QtCore.Qt.QueuedConnection)
        self.classifyThread.start()
        

    def finish_thread(self, res):
        '''
        Метод, вызывающийся после окончания выполнения потока
        '''
        prediction      = str(res["ImageNet class id"])
        predicted_class = str(res["Predicted class"])
        confidence      = str(round(res["Confidence"], 3))
        self.imageNetId.setText("ImageNet id: " + prediction)
        self.predictedClass.setText("Predicted class: " + predicted_class)
        self.confidence_label.setText("Confidence: " + confidence)
    
        
    def load_image(self, image_path):
        '''
        Метод загрузки изображения на QLabel

        Параметры:
            image_path (str): путь к изображению
        '''
        # создаем объект QPixmap из изображения по пути image_path
        pixmap = QPixmap(image_path)
        # ресайзим QPixmap до размеров лейбла imageLabel
        #pixmap.scaledToWidth(self.imageLabel.size().width())
        #pixmap.scaledToHeight(self.imageLabel.size().height())
        pixmap = pixmap.scaled(
                int(self.imageLabel.size().width()),
                int(self.imageLabel.size().height()),
                QtCore.Qt.KeepAspectRatio
            )
        # помещаем на лейбл ресайзнутое изображение
        self.imageLabel.setPixmap(pixmap)


    def aboutQt(self):
        '''
        Метод, выводящий окно сведений о Qt
        '''
        QtWidgets.QMessageBox.aboutQt(self, title="About Qt")


if __name__ == "__main__":
    # инициалзируем объект приложения и выводим его
    classifier_app = QtWidgets.QApplication(sys.argv)
    window_app     = ClassifierApp()
    window_app.show()
    # вызывается при закрытии приложения
    sys.exit(classifier_app.exec())