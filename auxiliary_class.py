# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 20:52:59 2020

@author: HP
"""
import numpy as np
from PyQt5.QtWidgets import QWidget, QSlider, QVBoxLayout, QHBoxLayout, \
    QLabel, QGridLayout, QProgressBar, QDesktopWidget, QLineEdit, QShortcut,\
    QPushButton, QComboBox, QMessageBox, QCheckBox, QListWidget, QListWidgetItem
import cv2
from PyQt5.QtCore import Qt, QBasicTimer, pyqtSignal, QEvent
from PyQt5.QtGui import QPixmap, QIcon, QIntValidator, QFont
import os
from autofinder.auxiliary_func import get_folder_from_file, np2qimage
from autofinder.stages.stage_rpi import Stage_Rpi
from threading import Thread

class QLabel(QLabel):
    def __init__(self, *args):
        super().__init__(*args)
        self.setFont(QFont('Book Antiqua', 10))

class QComboBox(QComboBox):
    def __init__(self, *args):
        super().__init__(*args)
        self.setFont(QFont('Book Antiqua', 10))

class QLineEdit(QLineEdit):
    def __init__(self, *args):
        super().__init__(*args)
        self.setFont(QFont('Book Antiqua', 10))

class DropLabel(QLabel):
    new_img = pyqtSignal(str)
    def __init__(self, *args):
        super().__init__(*args)
        self.setAcceptDrops(True)
        self.support_format = ['jpg', 'JPG', 'png', 'PNG', 'bmp', 'BMP']
        
    def dragEnterEvent(self, e):
        m = e.mimeData()
        if m.hasUrls():
            if m.urls()[0].toLocalFile()[-3:] in self.support_format:
                e.accept()
            else:
                e.ignore()
        else:
            e.ignore()
    
    def dropEvent(self, e):
        m = e.mimeData()
        if m.hasUrls():
            self.img_path = m.urls()[0].toLocalFile()
            self.img = cv2.imread(m.urls()[0].toLocalFile())
            self.new_img.emit('new') 
#            print(m.urls()[0].toLocalFile())
            

class RGB_Slider(QWidget):
    
    def __init__(self):
        super().__init__()
        self.initUI()
        
    
    def initUI(self):
        self.r_min_sld = QSlider(Qt.Horizontal,self)
        self.r_max_sld = QSlider(Qt.Horizontal,self)       
        self.g_min_sld = QSlider(Qt.Horizontal,self)
        self.g_max_sld = QSlider(Qt.Horizontal,self)        
        self.b_min_sld = QSlider(Qt.Horizontal,self)
        self.b_max_sld = QSlider(Qt.Horizontal,self)
        self.bright_sld = QSlider(Qt.Horizontal,self)
        self.contrast_sld = QSlider(Qt.Horizontal,self)
        
        bright_range = [-255, 255]
        contr_range = [0, 150]
        self.r_min_sld.setRange(*bright_range)
        self.r_max_sld.setRange(*contr_range)       
        self.g_min_sld.setRange(*bright_range)
        self.g_max_sld.setRange(*contr_range)
        self.b_min_sld.setRange(*bright_range)
        self.b_max_sld.setRange(*contr_range)
        self.bright_sld.setRange(*bright_range)
        self.contrast_sld.setRange(*contr_range)
        
        
        self.r_min_lbl = QLabel('R_bri', self)
        self.r_max_lbl = QLabel('R_contr', self)        
        self.g_min_lbl = QLabel('G_bri', self)
        self.g_max_lbl = QLabel('G_contr', self)       
        self.b_min_lbl = QLabel('B_bri', self)
        self.b_max_lbl = QLabel('B_contr', self)
        self.bright_lbl = QLabel('Bright', self)
        self.contrast_lbl = QLabel('Contrast', self)
        
        
        self.grid = QGridLayout()
        self.grid.addWidget(self.bright_lbl, *[1,0])
        self.grid.addWidget(self.bright_sld, *[1,1])
        
        self.grid.addWidget(self.contrast_lbl, *[2,0])
        self.grid.addWidget(self.contrast_sld, *[2,1])
        
        self.grid.addWidget(self.r_min_lbl, *[3,0])
        self.grid.addWidget(self.r_min_sld, *[3,1])
        
        self.grid.addWidget(self.r_max_lbl, *[4,0])
        self.grid.addWidget(self.r_max_sld, *[4,1])
        
        self.grid.addWidget(self.g_min_lbl, *[5,0])
        self.grid.addWidget(self.g_min_sld, *[5,1])
        
        self.grid.addWidget(self.g_max_lbl, *[6,0])
        self.grid.addWidget(self.g_max_sld, *[6,1])
        
        self.grid.addWidget(self.b_min_lbl, *[7,0])
        self.grid.addWidget(self.b_min_sld, *[7,1])
        
        self.grid.addWidget(self.b_max_lbl, *[8,0])
        self.grid.addWidget(self.b_max_sld, *[8,1])
          
        
        
class ProgressBar(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        self.pbar = QProgressBar(self)
        self.pbar.setGeometry(30, 40, 350, 25)

        self.timer = QBasicTimer()
        self.timer.start(50, self)
        self.progress = 0
        self.center()
        self.setWindowTitle('Capturing background...')
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        
    def timerEvent(self,e):
        self.pbar.setValue(self.progress)
    
    def center(self):
      qr = self.frameGeometry()
      cp = QDesktopWidget().availableGeometry().center()
      qr.moveCenter(cp)
      self.move(qr.topLeft())



class CameraNumEdit(QWidget):
    def __init__(self, current_num = 0):
        super().__init__()
        self.current_num = current_num
        self.camera_num = current_num
        self.init_ui()
        
    def init_ui(self):
        label_current = QLabel('Current number', self)
        self.label_current_value = QLabel(str(self.current_num), self)
        
        label_new = QLabel('New number: ', self)
        self.line_edit = QLineEdit(self)
        
        self.con_but_click_num = 0
        
        self.save_later_button = QPushButton('Save for later', self)
        self.save_later_button.clicked.connect(self.save)
        
        self.save_once_button = QPushButton('Save for once', self)
        self.save_once_button.clicked.connect(self.save)
        
        for sequence in ("Enter", "Return",):
            shorcut = QShortcut(sequence, self.save_once_button)
            shorcut.activated.connect(self.save_once_button.animateClick)
        
        cancel_button = QPushButton('Cancel', self)
        cancel_button.clicked.connect(self.cancel)
        
        hbox1 = QHBoxLayout()
        hbox1.addWidget(label_current)
        hbox1.addWidget(self.label_current_value)
        
        hbox2 = QHBoxLayout()
        hbox2.addWidget(label_new)
        hbox2.addWidget(self.line_edit)
        
        hbox3 = QHBoxLayout()
        hbox3.addWidget(self.save_later_button)
        hbox3.addWidget(self.save_once_button)
        hbox3.addWidget(cancel_button)
        
        vbox = QVBoxLayout()
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addLayout(hbox3)

        self.setLayout(vbox)
        self.setWindowTitle('Setting Camera Number')
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
    
    def save(self):
        self.camera_num = self.line_edit.text()
    
    def cancel(self):
        self.close()


        
        
        
        
        
        
class CalibrationEdit(QWidget):
    def __init__(self, current_calibration):
        super().__init__()
        self.current_calibration = current_calibration
        self.calibration = current_calibration
        self.init_ui()
        
    def init_ui(self):
#        label_default = QLabel(self)
#        label_default.setText('Default calibration')
#        label_default_value = QLabel(self)
#        label_default_value.setText('14.33')
        
        label_current = QLabel('Current calibration: ', self)
        self.label_current_value = QLabel('', self)
        self.label_current_value.setText(str(self.current_calibration))
        
        label_set = QLabel('New calibration: ', self)
        self.line_edit = QLineEdit(self)
        
        self.con_but_click_num = 0
        
        self.save_later_button = QPushButton('Save for later', self)
        self.save_later_button.clicked.connect(self.save_later)
        
        self.save_once_button = QPushButton('Save for once', self)
        self.save_once_button.clicked.connect(self.save_once)
        
        for sequence in ("Enter", "Return",):
            shorcut = QShortcut(sequence, self.save_once_button)
            shorcut.activated.connect(self.save_once_button.animateClick)
        
        cancel_button = QPushButton('Cancel', self)
        cancel_button.clicked.connect(self.cancel)
        
        hbox1 = QHBoxLayout()
        hbox1.addWidget(label_current)
        hbox1.addWidget(self.label_current_value)
        
        hbox2 = QHBoxLayout()
        hbox2.addWidget(label_set)
        hbox2.addWidget(self.line_edit)
        
        hbox3 = QHBoxLayout()
        hbox3.addWidget(self.save_later_button)
        hbox3.addWidget(self.save_once_button)
        hbox3.addWidget(cancel_button)
        
        vbox = QVBoxLayout()
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addLayout(hbox3)

        self.setLayout(vbox)
        self.setWindowTitle('Setting Calibration')
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
    
    def save_later(self):
        self.calibration = self.line_edit.text()
    
    def save_once(self):
        self.calibration = self.line_edit.text()
        if self.con_but_click_num == 0:
            self.con_but_click_num += 1
           
    def cancel(self):
        self.close()
        
        
        
        
        
class CustomContrast(QWidget):
    confirmed = pyqtSignal(str)
    def __init__(self, default_name):
        super().__init__()
        self.default_name = default_name
        print(default_name)
        self.init_ui()
        
    def init_ui(self):
        label_name = QLabel('Name: ', self)
        
        self.name_edit = QLineEdit(self)
        self.name_edit.setText(self.default_name)
        
        self.con_but_click_num = 0
        
        confirm_button = QPushButton('Confirm', self)
        confirm_button.clicked.connect(self.confirm)
        for sequence in ("Enter", "Return",):
            shorcut = QShortcut(sequence, confirm_button)
            shorcut.activated.connect(confirm_button.animateClick)
        
        cancel_button = QPushButton('Cancel', self)
        cancel_button.clicked.connect(self.cancel)
        
         
        hbox1 = QHBoxLayout()
        hbox1.addWidget(label_name)
        hbox1.addWidget(self.name_edit)
        
        hbox2 = QHBoxLayout()
        hbox2.addWidget(cancel_button)
        hbox2.addWidget(confirm_button)
        
        vbox = QVBoxLayout()
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)

        self.setLayout(vbox)
        self.setWindowTitle('Custom Contrast')
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        
    def confirm(self):
        if self.con_but_click_num == 0:
            self.con_but_click_num += 1
            
            self.name = self.name_edit.text()
            print(self.name)
            self.confirmed.emit('confirmed')    
   
    def cancel(self):
        self.close()     




class DeleteCustomContrast(QWidget):
    confirmed = pyqtSignal(str)
    def __init__(self, custom_contrast_list):
        super().__init__()
        self.custom_contrast_list = custom_contrast_list
        self.init_ui()
        
    def init_ui(self):
        self.qListWidget = QListWidget()
        
        
        self.check_box_list = []
        vbox = QVBoxLayout()
        for item in self.custom_contrast_list:
           self.check_box_list.append(QCheckBox(item, self))
           self.check_box_list[-1].setChecked(True)
           self.check_box_list[-1].toggle()
           
           qItem = QListWidgetItem(self.qListWidget)
           self.qListWidget.setItemWidget(qItem, self.check_box_list[-1])
#           vbox.addWidget(self.check_box_list[-1])
        
        vbox.addWidget(self.qListWidget)
        self.con_but_click_num = 0
        
        confirm_button = QPushButton('Confirm', self)
        confirm_button.clicked.connect(self.confirm)
        for sequence in ("Enter", "Return",):
            shorcut = QShortcut(sequence, confirm_button)
            shorcut.activated.connect(confirm_button.animateClick)
        
        cancel_button = QPushButton('Cancel', self)
        cancel_button.clicked.connect(self.cancel)
        
        hbox = QHBoxLayout()
        hbox.addWidget(cancel_button)
        hbox.addWidget(confirm_button)
        
        vbox.addLayout(hbox)
        
        self.setLayout(vbox)
        self.setWindowTitle('Delete Contrast')
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        
    def confirm(self):
        if self.con_but_click_num == 0:
            self.con_but_click_num += 1
            self.delte_list = []
            
            for i in range(len(self.custom_contrast_list)):
                if self.check_box_list[i].isChecked():
                    self.delte_list.append(self.custom_contrast_list[i])
            
            self.confirmed.emit('confirmed')
   
    def cancel(self):
        self.close()
           

        
class ThicknessChoose(QWidget):
    confirmed = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        label_material = QLabel('Material', self)
        # label_material.setText()
        
        label_thickness = QLabel('Thickness', self)
        # label_thickness.setText()
        
        self.combo_material = QComboBox(self)
        self.combo_material.addItem('graphene')
        self.combo_material.addItem('TMD')
        self.material = self.combo_material.currentText()
        
        self.combo_thickness = QComboBox(self)
        self.combo_thickness.addItem('285nm')
        self.combo_thickness.addItem('90nm')
        self.thickness = self.combo_thickness.currentText()
        
        self.con_but_click_num = 0
        confirm_button = QPushButton('Confirm', self)
        confirm_button.clicked.connect(self.confirm)
        for sequence in ("Enter", "Return",):
            shorcut = QShortcut(sequence, confirm_button)
            shorcut.activated.connect(confirm_button.animateClick)
        
        cancel_button = QPushButton('Cancel', self)
        cancel_button.clicked.connect(self.cancel)
        
        hbox1 = QHBoxLayout()
        hbox1.addWidget(label_material)
        hbox1.addWidget(self.combo_material)
        
        hbox2 = QHBoxLayout()
        hbox2.addWidget(label_thickness)
        hbox2.addWidget(self.combo_thickness)
        
        hbox3 = QHBoxLayout()
        hbox3.addWidget(confirm_button)
        hbox3.addWidget(cancel_button)
        
        vbox = QVBoxLayout()
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addLayout(hbox3)

        self.setLayout(vbox)
        
    def confirm(self):
        if self.con_but_click_num == 0:
            self.con_but_click_num += 1
            
            self.material = self.combo_material.currentText()
            self.thickness = self.combo_thickness.currentText()
            print(self.material, self.thickness)
            self.confirmed.emit('confirmed')
           
    def cancel(self):
        self.close()
        reply = QMessageBox.warning(self, "Warning", 'Do you want to search with '+\
                                    '285nm substrate?',+\
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            self.thickness_confirm.emit('285nm')
        else:
            reply = QMessageBox.warning(self, "Warning", 'Do you want to stop searching?',+\
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.No:
            self.show()
        
        
        
class SearchingProperty(QWidget):
    confirmed = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        label_material = QLabel('Material', self)
        
        label_thickness = QLabel('Substrate', self)
        label_mag = QLabel('Magnification', self)
        label_focus = QLabel('Focus method', self)
        label_size = QLabel('Scan area', self)
        label_contrast_min = QLabel('Thickness min (nm)', self)
        label_contrast_max = QLabel('Thickness max (nm)', self)
        label_target_size = QLabel('Flake size', self)
        
        self.combo_material = QComboBox(self)
        self.combo_material.addItem('Graphene')
        self.combo_material.addItem('hBN')
        self.combo_material.addItem('WS2')
        self.combo_material.addItem('WSe2')
        self.material = self.combo_material.currentText()
        
        self.combo_substrate = QComboBox(self)
        # self.combo_thickness.addItem('285nm')
        self.combo_substrate.addItem('90 nm')
        self.substrate = self.combo_substrate.currentText()
        
        self.combo_mag = QComboBox(self)
        # self.combo_mag.addItem('5x')
        self.combo_mag.addItem('10x')
        # self.combo_mag.addItem('20x')
        self.magnification = self.combo_mag.currentText()
        
        self.combo_focus = QComboBox(self)
        self.combo_focus.addItem('Every point')
        # self.combo_focus.addItem('3 points plane')
        self.focus_method = self.combo_focus.currentText()
        
        self.combo_size = QComboBox(self)
        self.combo_size.addItem('1 cm * 1 cm')
        self.combo_size.addItem('2 cm * 2 cm')
        self.combo_size.addItem('3 cm * 3 cm')
        self.combo_size.addItem('4 cm * 4 cm')
        self.combo_size.addItem('5 cm * 5 cm')
        self.combo_size.addItem('6 cm * 6 cm')
        self.combo_size.addItem('7 cm * 7 cm')
        self.scan_area = self.combo_size.currentText()
        
        self.combo_thickness_min = QComboBox(self)
        for i in range(51):
            self.combo_thickness_min.addItem(str(i) + ' nm')
        self.thickness_min = self.combo_thickness_min.currentText()
        
        self.combo_thickness_max = QComboBox(self)
        for i in range(51):
            self.combo_thickness_max.addItem(str(i) + ' nm')
        self.combo_thickness_max.setCurrentText('50 nm')
        self.thickness_max = self.combo_thickness_max.currentText()
        
        self.combo_target_size = QComboBox(self)
        self.combo_target_size.addItem('10 * 10 um^2')
        self.combo_target_size.addItem('20 * 20 um^2')
        self.combo_target_size.addItem('30 * 30 um^2')
        self.combo_target_size.addItem('40 * 40 um^2')
        self.combo_target_size.setCurrentText('20 * 20 um^2')
        self.flake_size = self.combo_target_size.currentText()
        
        self.con_but_click_num = 0
        confirm_button = QPushButton('Confirm', self)
        confirm_button.clicked.connect(self.confirm)
        for sequence in ("Enter", "Return",):
            shorcut = QShortcut(sequence, confirm_button)
            shorcut.activated.connect(confirm_button.animateClick)
        
        cancel_button = QPushButton('Cancel', self)
        cancel_button.clicked.connect(self.cancel)
        
        hbox_material = QHBoxLayout()
        hbox_material.addWidget(label_material)
        hbox_material.addWidget(self.combo_material)
        
        hbox_thickness = QHBoxLayout()
        hbox_thickness.addWidget(label_thickness)
        hbox_thickness.addWidget(self.combo_substrate)
        
        hbox_mag = QHBoxLayout()
        hbox_mag.addWidget(label_mag)
        hbox_mag.addWidget(self.combo_mag)
        
        hbox_focus = QHBoxLayout()
        hbox_focus.addWidget(label_focus)
        hbox_focus.addWidget(self.combo_focus)
        
        hbox_size = QHBoxLayout()
        hbox_size.addWidget(label_size)
        hbox_size.addWidget(self.combo_size)
        
        hbox_contrast_min = QHBoxLayout()
        hbox_contrast_min.addWidget(label_contrast_min)
        hbox_contrast_min.addWidget(self.combo_thickness_min)

        hbox_contrast_max = QHBoxLayout()
        hbox_contrast_max.addWidget(label_contrast_max)
        hbox_contrast_max.addWidget(self.combo_thickness_max)
        
        hbox_target_size = QHBoxLayout()
        hbox_target_size.addWidget(label_target_size)
        hbox_target_size.addWidget(self.combo_target_size)
        
        hbox_confirm = QHBoxLayout()
        hbox_confirm.addWidget(confirm_button)
        hbox_confirm.addWidget(cancel_button)
        
        vbox = QVBoxLayout()
        vbox.addLayout(hbox_material)
        vbox.addLayout(hbox_thickness)
        vbox.addLayout(hbox_mag)
        vbox.addLayout(hbox_focus)
        vbox.addLayout(hbox_size)
        vbox.addLayout(hbox_contrast_min)
        vbox.addLayout(hbox_contrast_max)
        vbox.addLayout(hbox_target_size)
        vbox.addLayout(hbox_confirm)
        
        self.setLayout(vbox)
        self.setWindowTitle('Search setting')
        self.resize(370,400)
        
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        
    def confirm(self):
        if self.con_but_click_num == 0:
            self.con_but_click_num += 1
            
            self.material = self.combo_material.currentText()
            self.substrate = self.combo_substrate.currentText()
            self.magnification = self.combo_mag.currentText()
            self.focus_method = self.combo_focus.currentText()
            self.scan_area = self.combo_size.currentText()
            self.flake_size = self.combo_target_size.currentText()
            self.thickness_min = self.combo_thickness_min.currentText()
            self.thickness_max = self.combo_thickness_max.currentText()
            print(self.material, self.substrate, self.magnification, self.focus_method,
                  self.scan_area, self.thickness_min, self.thickness_max, self.flake_size)
            self.confirmed.emit('confirmed')
            
    def cancel(self):
        self.close()
#        reply = QMessageBox.warning(self, "Warning", 'Do you want to search with '+\
#                                    '285nm substrate?',+\
#                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
#        if reply == QMessageBox.Yes:
#            self.thickness_confirm.emit('285nm')
#        else:
#            reply = QMessageBox.warning(self, "Warning", 'Do you want to stop searching?',+\
#                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
#        if reply == QMessageBox.No:
#            self.show()
            
        
class MovePanel(QWidget):
    move_signal = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        self.stage = Stage_Rpi()
        self.stage.initialize()
#        self.move_pos = [0, 0, 0]
        
        self.current_dir = os.path.abspath(__file__).replace('\\','/')
        self.current_dir = get_folder_from_file(self.current_dir)
        self.current_dir = self.current_dir + 'support_file/'
        
        self.move_forward_button = QPushButton(self)
        self.move_forward_button.setIcon(QIcon(self.current_dir+'array_key_up.png'))
        self.move_forward_button.pressed.connect(self.move_forward)
        self.move_forward_button.released.connect(self.stop_stage)
        
        self.move_upward_button = QPushButton(self)
        self.move_upward_button.setIcon(QIcon(self.current_dir+'array_key_up.png'))
        self.move_upward_button.pressed.connect(self.move_upward)
        self.move_upward_button.released.connect(self.stop_stage)
        
        self.move_leftward_button = QPushButton(self)
        self.move_leftward_button.setIcon(QIcon(self.current_dir+'array_key_left.png'))
        self.move_leftward_button.pressed.connect(self.move_leftward)
        self.move_leftward_button.released.connect(self.stop_stage)
        
        self.move_rightward_button = QPushButton(self)
        self.move_rightward_button.setIcon(QIcon(self.current_dir+'array_key_right.png'))
        self.move_rightward_button.pressed.connect(self.move_rightward)
        self.move_rightward_button.released.connect(self.stop_stage)
        
        self.move_backward_button = QPushButton(self)
        self.move_backward_button.setIcon(QIcon(self.current_dir+'array_key_down.png'))
        self.move_backward_button.pressed.connect(self.move_backward)
        self.move_backward_button.released.connect(self.stop_stage)
        
        self.move_downward_button = QPushButton(self)
        self.move_downward_button.setIcon(QIcon(self.current_dir+'array_key_down.png'))
        self.move_downward_button.pressed.connect(self.move_downward)
        self.move_downward_button.released.connect(self.stop_stage)
        
        # self.stop_button = QPushButton('Stop', self)
        # self.stop_button.clicked.connect(self.emergency_stop)
        
        # self.restore_button = QPushButton('Restore', self)
        # self.restore_button.clicked.connect(self.restore_stage)
        
        # self.continuous_cb = QCheckBox('Continuous', self)
        # self.continuous_cb.stateChanged.connect(self.set_continuity)
        self.continuous = False
        
        self.onlyInt = QIntValidator()
        spacing_label = QLabel('   ', self)
        
        self.x_step = 1000
        self.y_step = 1000
        self.z_step = 1000
        
        self.x_speed = 1
        self.y_speed = 1
        self.z_speed = 1
        
        self.x_label = QLabel('X step', self)
        self.x_edit = QLineEdit(str(self.x_step), self)
        self.x_edit.setValidator(self.onlyInt)
        self.x_edit.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.x_edit.setFixedWidth(70)
        self.x_unit = QLabel('um', self)
        
        self.y_label = QLabel('Y step', self)
        self.y_edit = QLineEdit(str(self.y_step), self)
        self.y_edit.setValidator(self.onlyInt)
        self.y_edit.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.y_edit.setFixedWidth(70)
        self.y_unit = QLabel('um', self)
        
        self.z_label = QLabel('Z step', self)
        self.z_edit = QLineEdit(str(self.z_step), self)
        self.z_edit.setValidator(self.onlyInt)
        self.z_edit.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.z_edit.setFixedWidth(70)
        self.z_unit = QLabel('', self)
        
        
        self.grid = QGridLayout()
        self.grid.addWidget(self.move_forward_button, *[1,1])
        self.grid.addWidget(spacing_label, *[1,3])
        self.grid.addWidget(self.move_upward_button, *[1,4])
        self.grid.addWidget(spacing_label, *[1,5])
        self.grid.addWidget(self.x_label, *[1,6])
        self.grid.addWidget(self.x_edit, *[1,7])
        self.grid.addWidget(self.x_unit, *[1,8])
        self.grid.addWidget(spacing_label, *[1,9])
        # self.grid.addWidget(self.stop_button, *[1,10])
        
        self.grid.addWidget(self.move_leftward_button, *[2,0])
        self.grid.addWidget(self.move_rightward_button, *[2,2])
        self.grid.addWidget(self.y_label, *[2,6])
        self.grid.addWidget(self.y_edit, *[2,7])
        self.grid.addWidget(self.y_unit, *[2,8])
        # self.grid.addWidget(self.restore_button, *[2,10])
        
        self.grid.addWidget(self.move_backward_button, *[3,1])
        self.grid.addWidget(self.move_downward_button, *[3,4])
        self.grid.addWidget(self.z_label, *[3,6])
        self.grid.addWidget(self.z_edit, *[3,7])
        self.grid.addWidget(self.z_unit, *[3,8])
        # self.grid.addWidget(self.continuous_cb, *[3,10])
        
        self.setLayout(self.grid)
        print('width, height',self.width(), self.height())
        self.setFixedSize(550, 120)
        self.setWindowTitle('Move stage')
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.setMouseTracking(True)
        
        
    def set_continuity(self, state):
        if state == Qt.Checked:
#            self.cb_tool_distance.setChecked(True)
            self.continuous = True
        else:
            self.continuous = False
        self.change_label()
        print('continuous', self.continuous)
        
    def change_label(self):
        if self.continuous:
            self.x_label.setText('X speed')
            self.y_label.setText('Y speed')
            self.z_label.setText('Z speed')
            self.x_edit.setText(str(self.x_speed))
            self.y_edit.setText(str(self.y_speed))
            self.z_edit.setText(str(self.z_speed))
            self.x_unit.setText('mm/s')
            self.y_unit.setText('mm/s')
            self.z_unit.setText('mm/s')
        else:
            self.x_label.setText('X step')
            self.y_label.setText('Y step')
            self.z_label.setText('Z step')
            self.x_edit.setText(str(self.x_step))
            self.y_edit.setText(str(self.y_step))
            self.z_edit.setText(str(self.z_step))
            self.x_unit.setText('um')
            self.y_unit.setText('um')
            # self.z_unit.setText('um')
            
    def stop_stage(self):
        if self.continuous:
            print('terminate')
            self.stage.terminate()
        
    def move_forward(self):
        self.validate_speed_and_step()
        if self.continuous:
            self.stage.thread_move_pos = [0, 50000, 0, int(self.y_edit.text())]
        else:
            temp = Thread(target = self.stage.move_xyz, args = (0, int(self.y_edit.text()), 0))
            temp.start()
        
    def move_backward(self):
        self.validate_speed_and_step()
        if self.continuous:
            self.stage.thread_move_pos = [0, -50000, 0, int(self.y_edit.text())]
        else:
            temp = Thread(target = self.stage.move_xyz, args = (0, -int(self.y_edit.text()), 0))
            temp.start()
        
    def move_leftward(self):
        self.validate_speed_and_step()
        if self.continuous:
            self.stage.thread_move_pos = [50000, 0, 0, int(self.x_edit.text())]
        else:
            temp = Thread(target = self.stage.move_xyz, args = (int(self.x_edit.text()), 0, 0))
            temp.start()
        
    def move_rightward(self):
        self.validate_speed_and_step()
        if self.continuous:
            self.stage.thread_move_pos = [-50000, 0, 0, int(self.x_edit.text())]
        else:
            temp = Thread(target = self.stage.move_xyz, args = (-int(self.x_edit.text()), 0, 0))
            temp.start()
        
    def move_upward(self):
        self.validate_speed_and_step()
        if self.continuous:
            self.stage.thread_move_pos = [0, 0, 10000, int(self.z_edit.text())]
        else:
            temp = Thread(target = self.stage.move_xyz, args = (0, 0, int(self.z_edit.text())))
            temp.start()
        
    def move_downward(self):
        self.validate_speed_and_step()
        if self.continuous:
            self.stage.thread_move_pos = [0, 0, -10000, int(self.z_edit.text())]
        else:
            temp = Thread(target = self.stage.move_xyz, args = (0, 0, -int(self.z_edit.text())))
            temp.start()
        
    def validate_speed_and_step(self):
        if self.continuous:
            self.y_speed = self.test_speed(int(self.y_edit.text()))
            self.y_edit.setText(str(self.y_speed))
            
            self.x_speed = self.test_speed(int(self.x_edit.text()))
            self.x_edit.setText(str(self.x_speed))
            
            self.z_speed = self.test_speed(int(self.z_edit.text()))
            self.z_edit.setText(str(self.z_speed))
        else:
            self.y_step = int(self.y_edit.text())
            self.x_step = int(self.x_edit.text())
            self.z_step = int(self.z_edit.text())
        
    def test_speed(self, speed):
        if speed < 1:
            speed = 1
        elif speed >10:
            speed = 10
        return speed
    
    def emergency_stop(self):
        self.stage.terminate()
        
    def restore_stage(self):
        self.stage.restore()
        
        

    

class AutoIsoTv(QWidget):

    def __init__(self):
            super().__init__()
            self.init_ui()
            
    def init_ui(self):
        ISO_list = ['None', 'AUTO', '100', '125', '160', '200', '250', '320', '400',
                    '500', '640', '800', '1000', '1250', '1600', '2000', '2500', '3200',
                    '4000', '5000', '6400', '8000', '10000', '12800', '16000', '20000',
                    '25600', '32000', '40000', '51200']
        
        Tv_list = ['None', '30"', '25"', '20"', '15"', '13"', '10"',
                   '8"', '6"', '5"', '4"', '3.2"', '2.5"', '2"', '1.6"', '1.3"', '1"', '0.8"',
                   '0.6"', '0.5"', '0.4"', '1/3', '1/4', '1/5', '1/6', '1/8', '1/10', '1/13',
                   '1/15', '1/20', '1/25', '1/30', '1/40', '1/50', '1/60', '1/80', '1/100', '1/125',
                   '1/160', '1/200', '1/250', '1/320', '1/400', '1/500', '1/640', '1/800',
                   '1/1000', '1/1250', '1/1600', '1/2000', '1/2500', '1/3200', '1/4000']
        
        label_mag_list = [QLabel('  5x', self), QLabel(' 10x', self), QLabel(' 20x', self), 
                          QLabel(' 60x', self), QLabel('100x', self)]
        
        label_ISO_list = [QLabel('ISO', self) for i in range(5)]
        label_Tv_list = [QLabel('Tv', self) for i in range(5)]
        
        self.ISO_combo_dict = {'5x': QComboBox(self), '10x': QComboBox(self), '20x': QComboBox(self),
                               '60x': QComboBox(self), '100x': QComboBox(self)}
        
        self.Tv_combo_dict = {'5x': QComboBox(self), '10x': QComboBox(self), '20x': QComboBox(self),
                               '60x': QComboBox(self), '100x': QComboBox(self)}
        
        mag_list = ['5x', '10x', '20x', '60x', '100x']
        
        hbox_list = [QHBoxLayout() for i in range(5)]
        
        for i in range(5):
            hbox_list[i].addWidget(label_mag_list[i])
            
            hbox_list[i].addStretch(1)
            hbox_list[i].addWidget(label_ISO_list[i])
            hbox_list[i].addWidget(self.ISO_combo_dict[mag_list[i]])
            self.ISO_combo_dict[mag_list[i]].addItems(ISO_list)
            self.ISO_combo_dict[mag_list[i]].setFixedWidth(80)
            
            hbox_list[i].addStretch(1)
            hbox_list[i].addWidget(label_Tv_list[i])
            hbox_list[i].addWidget(self.Tv_combo_dict[mag_list[i]])
            self.Tv_combo_dict[mag_list[i]].addItems(Tv_list)
            self.Tv_combo_dict[mag_list[i]].setFixedWidth(80)
        
        self.con_but_click_num = 0
        
        self.confirm_button = QPushButton('Confirm', self)
        self.confirm_button.clicked.connect(self.confirmed)
        
        for sequence in ("Enter", "Return",):
            shorcut = QShortcut(sequence, self.confirm_button)
            shorcut.activated.connect(self.confirm_button.animateClick)
        
        cancel_button = QPushButton('Cancel', self)
        cancel_button.clicked.connect(self.cancel)
        
        hbox_confirm = QHBoxLayout()
        hbox_confirm.addWidget(self.confirm_button)
        hbox_confirm.addWidget(cancel_button)
        
        vbox = QVBoxLayout()
        for i in range(5):
            vbox.addLayout(hbox_list[i])
        vbox.addLayout(hbox_confirm)

        self.setLayout(vbox)
        self.setWindowTitle('Setting Auto ISO and Tv')
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
    
    def save_later(self):
        self.calibration = self.line_edit.text()
    
    def save_once(self):
        self.calibration = self.line_edit.text()
        if self.con_but_click_num == 0:
            self.con_but_click_num += 1
            
    def confirmed(self):
        self.close()
           
    def cancel(self):
        self.close()
    
    
    
    