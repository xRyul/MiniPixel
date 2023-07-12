from io import BytesIO
import os
import sys
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QEvent, QThread, pyqtSignal
from PyQt5.QtGui import QDragEnterEvent, QDropEvent
from PyQt5.QtWidgets import (QFileDialog, QVBoxLayout, QHBoxLayout,
                             QPushButton, QListWidget, QProgressBar,
                             QCheckBox, QLabel, QSpacerItem,
                             QSizePolicy)

from PIL import Image
import mozjpeg_lossless_optimization
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from psd_tools import PSDImage

class WorkerThread(QThread):
    progress_signal = pyqtSignal(int)

    def __init__(self, file_paths, process_image):
        QThread.__init__(self)
        self.file_paths = file_paths
        self.process_image = process_image

    def run(self):
        try:
            print('WorkerThread started')
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.process_image, file_path) for file_path in self.file_paths]
                for i, future in enumerate(as_completed(futures)):
                    future.result()
                    progress = int((i + 1) / len(self.file_paths) * 100)
                    self.progress_signal.emit(progress)
        except Exception as e:
            logging.error(f'Error processing images: {e}')

class CustomTitleBar(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Add a label to display the window title
        self.title_label = QLabel(self.parent.windowTitle())
        layout.addWidget(self.title_label)

        # Add spacer item to push buttons to the right
        layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Fixed))

        # Add close button
        self.close_button = QPushButton('✕')
        self.close_button.clicked.connect(self.parent.close)
        self.close_button.setFixedSize(35, 35)
        self.close_button.setObjectName("closeButton")

        layout.addWidget(self.close_button)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.parent.mouse_press_pos = event.globalPos()
            self.parent.mouse_press_window_pos = self.parent.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            move_pos = event.globalPos() - self.parent.mouse_press_pos
            new_pos = self.parent.mouse_press_window_pos + move_pos
            self.parent.move(new_pos)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.toggle_maximize()

class FileListWidget(QListWidget):
    def __init__(self, parent=None):
        super(FileListWidget, self).__init__(parent)
        self.setAcceptDrops(True)

        self.clear_button = QPushButton('Clear')
        self.clear_button.setObjectName("clearButton")
        self.clear_button.setFixedSize(40, 25)
        self.clear_button.move(10, 10)
        self.clear_button.show()
        self.clear_button.raise_()
        self.clear_button.setParent(self.viewport())
        self.clear_button.clicked.connect(self.clear)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        clear_button_pos = (self.viewport().width() - 45, 5)
        self.clear_button.move(*clear_button_pos)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                self.addItem(url.toLocalFile())
            event.acceptProposedAction()

class MozJPEGGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super(MozJPEGGUI, self).__init__()

        self.setAttribute(Qt.WA_TranslucentBackground)
        # Create frame to cover entire window
        self.main_frame = QtWidgets.QFrame(self)
        self.main_frame.setGeometry(self.rect())
        self.main_frame.setStyleSheet('''
            QFrame {
                border-radius: 10px;
                background-color: #282a36;
            }
        ''')

        # Remove default title bar and window frame
        self.setWindowFlags(Qt.FramelessWindowHint)

        # Create custom title bar
        self.title_bar = CustomTitleBar(self)
        self.setMenuWidget(self.title_bar)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)

        self.select_button = QPushButton('Select Images')
        button_layout.addWidget(self.select_button)
        self.select_button.clicked.connect(self.select_images)

        self.process_folder_checkbox = QCheckBox('Process Sub-folders')
        button_layout.addWidget(self.process_folder_checkbox)
        self.file_list = FileListWidget()
        layout.addWidget(self.file_list)

        # Output Button and Label
        output_layout = QHBoxLayout()
        layout.addLayout(output_layout)
        self.output_label = QLabel('Output to: ')
        self.output_dropdown = QtWidgets.QComboBox()
        self.output_dropdown.addItems(['Source folder', 'Custom location'])
        
        output_layout.addWidget(self.output_label)
        output_layout.addWidget(self.output_dropdown)
        self.output_dropdown.currentIndexChanged.connect(self.output_changed)

        self.optimize_button = QPushButton('Optimize')
        layout.addWidget(self.optimize_button)
        self.optimize_button.clicked.connect(self.optimize_images)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        self.progress_bar.hide() # Hide progress bar initially

        self.status_label = QLabel()
        layout.addWidget(self.status_label)

        def changeEvent(self, event):
            if event.type() == QEvent.WindowTitleChange:
                # Update title label when window title changes
                self.title_bar.title_label.setText(self.windowTitle())

        def mousePressEvent(self, event):
            # Initialize mouse press position variables
            self.mouse_press_pos = None
            self.mouse_press_window_pos = None

        self.show()

    def calculate_psnr(self, img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    
    def adjust_quality(self, img, desired_psnr):
        quality = 100
        while True:
            with BytesIO() as buffer:
                img.save(buffer, format='JPEG', quality=quality)
                compressed_img = Image.open(buffer)
                psnr = self.calculate_psnr(np.array(img), np.array(compressed_img))
                # print(f'Quality: {quality}, PSNR: {psnr}')
                if psnr >= desired_psnr or quality <= 0:
                    break
                quality -= 1
        return max(quality, 0) if psnr >= desired_psnr else 99
        
    def select_images(self):
        if self.process_folder_checkbox.isChecked():
            dir_name = QFileDialog.getExistingDirectory(self, 'Select Folder')
            if dir_name:
                self.file_list.addItem(dir_name)
                self.file_list.item(0).setFlags(Qt.NoItemFlags)
                for i in range(1, self.file_list.count()):
                    self.file_list.takeItem(i)
                self.file_list.setDragEnabled(False)
                self.file_list.clear_button.show()
                QtWidgets.QApplication.processEvents()
                clear_button_pos = (self.file_list.viewport().width() - 60,
                                    10)
                self.file_list.clear_button.move(*clear_button_pos)
                QtWidgets.QApplication.processEvents()
            else:
                return
        else:
            file_names, _ = QFileDialog.getOpenFileNames(self, 'Select Images')
            if file_names:
                self.file_list.addItems(file_names)
            else:
                return
            
    def output_changed(self, index):
        if index == 1: # Custom location
            dir_name = QFileDialog.getExistingDirectory(self, 'Select Output Directory')
            if dir_name:
                self.output_label.setText(f'Output to: {dir_name}')
            else:
                self.output_dropdown.setCurrentIndex(0) # Reset to Source folder

    def process_image(self, file_path):
        try:
            if file_path.lower().endswith('.psd'):
                psd = PSDImage.open(file_path)
                image = psd.composite()
            else:
                image = Image.open(file_path)
                
            with image:
                original_icc_profile = image.info.get('icc_profile')
                quality = self.adjust_quality(image, 45)
                with BytesIO() as buffer:
                    if image.mode in ('RGBA', 'LA'):
                        image = image.convert('RGB')
                    image.save(buffer, format='JPEG', quality=quality)
                    output_bytes = mozjpeg_lossless_optimization.optimize(buffer.getvalue())

            if self.output_dropdown.currentIndex() == 0: # Source folder
                dir_path, file_name = os.path.split(file_path)
                base_dir = self.file_list.item(0).text()
                rel_dir = os.path.relpath(dir_path, base_dir)
                parent_dir = os.path.dirname(base_dir)
                output_dir = os.path.join(parent_dir, 'mozJPEG', os.path.basename(base_dir), rel_dir)
            else: # Custom location
                output_dir = self.output_label.text().replace('Output to: ', '')
                _, file_name = os.path.split(file_path)

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = os.path.join(output_dir,
                                    os.path.splitext(file_name)[0] + '.jpg')

            with open(output_path, 'wb') as output_file:
                output_file.write(output_bytes)

            with Image.open(output_path) as optimized_image:
                optimized_image.save(output_path, icc_profile=original_icc_profile)
        except Exception as e:
            logging.error(f'Error processing image {file_path}: {e}')

    def optimize_images(self):
        try:
            print('Optimize button clicked')
            logging.info('Optimize button clicked')
            self.optimize_button.setText('Stop') # Change text to Stop
            self.optimize_button.clicked.disconnect() # Disconnect previous clicked signal
            self.optimize_button.clicked.connect(self.stop_optimization) # Connect new clicked signal to stop_optimization method
            if self.process_folder_checkbox.isChecked():
                dir_path = self.file_list.item(0).text()
                print(f'Selected folder: {dir_path}')
                logging.info(f'Selected folder: {dir_path}')
                file_paths = []
                for root, dirs, files in os.walk(dir_path):
                    for file_name in files:
                        if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.psd')):
                            file_path = os.path.join(root, file_name)
                            print(f'Found file: {file_path}')
                            logging.info(f'Found file: {file_path}')
                            file_paths.append(file_path)
            else:
                num_files = self.file_list.count()
                file_paths = [self.file_list.item(i).text() for i in range(num_files)]

            print(f'File paths: {file_paths}')
            logging.info(f'File paths: {file_paths}')

            self.worker_thread = WorkerThread(file_paths, self.process_image)
            self.worker_thread.progress_signal.connect(self.update_progress_bar)
            self.worker_thread.start()
        except Exception as e:
            print(f'Error in optimize_images: {e}')
            logging.error(f'Error in optimize_images: {e}')

    def update_progress_bar(self, progress):
        # Set background color to match Dracula theme
        background_color = "#282a36"
        # Set progress color to match Dracula theme
        progress_color = "#bd93f9"
        # Create a linear gradient with the background and progress colors
        gradient = f"qlineargradient(x1:0, y1:0.5, x2:1, y2:0.5, stop:0 {progress_color}, stop:{progress/100} {progress_color}, stop:{progress/100} {background_color}, stop:1 {background_color})"
        # Update the style sheet of the Optimize button with the new gradient and border radius
        self.optimize_button.setStyleSheet(f"background: {gradient}; border-radius: 15px;")
        if progress == 100:
            self.optimize_button.setText('Done') # Show Done message in the button

    def stop_optimization(self):
        if self.worker_thread.isRunning():
            self.worker_thread.terminate()
        self.optimize_button.setText('Optimize') # Reset text to Optimize
        self.optimize_button.setStyleSheet("") # Reset style sheet
        self.optimize_button.clicked.disconnect() # Disconnect previous clicked signal
        self.optimize_button.clicked.connect(self.optimize_images) # Connect new clicked signal to optimize_images method

    def output_images(self):
        dir_name = QFileDialog.getExistingDirectory(self, 'Select Output Directory')
        if dir_name:
            self.output_label.setText(f'Output Directory: {dir_name}')
        else:
            return
        

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    with open("style.qss", "r") as f:
        _style = f.read()
        app.setStyleSheet(_style)
    window = MozJPEGGUI()
    app.exec_()
