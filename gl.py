import hashlib
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QPushButton, QLabel, QVBoxLayout, QMessageBox, QFileDialog, QTableWidget, QTableWidgetItem
import sqlite3
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow import keras

def hash_password(password):
    salt = b'mysalt'
    key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return key

class LoginWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Авторизация")
        self.setGeometry(100, 100, 300, 200)
        self.username_label = QLabel("Имя пользователя:")
        self.username_input = QLineEdit()

        self.password_label = QLabel("Пароль:")
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)

        self.login_button = QPushButton("Войти")
        self.login_button.clicked.connect(self.login)

        self.register_button = QPushButton("Регистрация")
        self.register_button.clicked.connect(self.reg)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.username_label)
        self.layout.addWidget(self.username_input)
        self.layout.addWidget(self.password_label)
        self.layout.addWidget(self.password_input)
        self.layout.addWidget(self.login_button)
        self.layout.addWidget(self.register_button)

        self.setLayout(self.layout)


    def login(self):
        username = self.username_input.text()
        password = self.password_input.text()
        password1 = hash_password(password)
        conn = sqlite3.connect('bb.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE login=? AND password=?", (username, password1))
        result = cursor.fetchone()

        if result:
            QMessageBox.information(self, "Успешная авторизация", "Вы успешно вошли в систему, как пользователь!")
            self.user_dashboard = UserDashboardWindow(username)
            self.user_dashboard.show()
            self.close()
        else:
            cursor.execute("SELECT * FROM admins WHERE login=? AND password=?", (username, password))
            result = cursor.fetchone()

            if result:
                QMessageBox.information(self, "Успешная авторизация", "Вы успешно вошли в систему, как администратор!")
                self.apanel = apanel()
                self.apanel.show()
            else:
                QMessageBox.warning(self, "Ошибка авторизации", "Неверное имя пользователя или пароль")

        conn.close()

    def reg(self):
        self.registration_window = RegistrationWindow()  # Сохраняем ссылку на объект UserDashboardWindow
        self.registration_window.show()

    def openUserDashboard(self):
        user_dashboard = UserDashboardWindow()
        user_dashboard.show()

    def openRegistrationWindow(self):
        registration_window = RegistrationWindow()
        registration_window.show()


class UserDashboardWindow(QWidget):
    def __init__(self, username):
        super().__init__()

        self.setWindowTitle("Личный кабинет")
        self.setGeometry(200, 200, 400, 300)

        self.lbl2 = QLabel("Добро пожаловать, " + username, self)
        self.lbl2.move(10, 10)

        self.image_path_label = QLabel("Путь к изображению:")
        self.image_path_input = QLineEdit()
        self.browse_button = QPushButton("Обзор")
        self.browse_button.clicked.connect(self.browseImage)

        self.upload_button = QPushButton("Определить эмоциональное состояние")
        self.upload_button.clicked.connect(self.uploadImage)

        self.ropr = QPushButton("Показать раннее определённые состояния")
        self.ropr.clicked.connect(self.rann)

        self.image_label = QLabel()

        self.lexit = QPushButton("Выйти")
        self.lexit.clicked.connect(self.close)
        self.prr = QLabel(self)
        self.massopr = QLabel(self)
        layout = QVBoxLayout()
        layout.addWidget(self.lbl2)
        layout.addWidget(self.image_path_label)
        layout.addWidget(self.image_path_input)
        layout.addWidget(self.browse_button)
        layout.addWidget(self.upload_button)
        layout.addWidget(self.image_label)
        layout.addWidget(self.prr)
        layout.addWidget(self.ropr)
        layout.addWidget(self.massopr)
        layout.addWidget(self.lexit)
        self.setLayout(layout)
        self.mmem = [-1, -1, -1]
        self.kz = 0
    def browseImage(self):
        self.prr.setText("")
        filename, _ = QFileDialog.getOpenFileName(self, "Выберите изображение", "", "Image Files (*.png *.jpg *.jpeg)")
        if filename:
            self.image_path_input.setText(filename)
            pixmap = QPixmap(filename)
            self.image_label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio))

    def uploadImage(self):
        model = keras.models.load_model('mm111.keras')
        path = self.image_path_input.text()

        def predict_image(img_path, model):
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            prediction = model.predict(img_array)

            labels = ["гнев", "отвращение", "страх", "радость", "нейтральное состояние", "грусть", "удивление"]

            self.predicted_class_index = np.argmax(prediction)
            self.prr.setText("На изображении: " + labels[self.predicted_class_index])

        predict_image(path, model)

        if self.mmem[self.kz] == -1:
            self.mmem[self.kz] = self.predicted_class_index
            labels = ["гнев", "отвращение", "страх", "радость", "нейтральное состояние", "грусть", "удивление"]
            self.mmem[self.kz] = labels[self.predicted_class_index]

        if self.kz < len(self.mmem) - 1:
            self.kz = self.kz + 1
        else:
            QMessageBox.information(self, "Масимальное кол-во тестирований", "Следующие тестирования не будут сохранены!")

    def rann(self):
        if self.mmem[0] != -1 or self.mmem[1] != -1 or self.mmem[2] != -1:
            array_string = ', '.join(str(item) for item in self.mmem)
            self.massopr.setText("Раннее определённые состояния: " + array_string)
        else:
            QMessageBox.information(self, "Ошибка", "Нет раннее определённых состояний!")


class RegistrationWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Регистрация")
        self.setGeometry(200, 200, 400, 300)

        self.username_label = QLabel("Имя пользователя:")
        self.username_input = QLineEdit()

        self.password_label = QLabel("Пароль:")
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)

        self.FIO_label = QLabel("ФИО:")
        self.FIO_input = QLineEdit()
        self.group_label = QLabel("Группа:")
        self.group_input = QLineEdit()

        register_button = QPushButton("Регистрация")
        register_button.clicked.connect(self.regg)

        layout = QVBoxLayout()
        layout.addWidget(self.username_label)
        layout.addWidget(self.username_input)
        layout.addWidget(self.password_label)
        layout.addWidget(self.password_input)
        layout.addWidget(self.FIO_label)
        layout.addWidget(self.FIO_input)
        layout.addWidget(self.group_label)
        layout.addWidget(self.group_input)
        layout.addWidget(register_button)

        self.setLayout(layout)


    def regg(self):
        username = self.username_input.text()
        password = self.password_input.text()
        FIO = self.FIO_input.text()
        gr = self.group_input.text()
        password = hash_password(password)
        conn = sqlite3.connect('bb.db')
        cursor = conn.cursor()
        try:
            cursor.execute('INSERT INTO users (login, password, FIO, gr) VALUES (?, ?, ?, ?)', (username, password, FIO, gr))
            conn.commit()
            QMessageBox.information(self, "Успешная регистрация", "Вы успешно зарегистрировались!")
            self.close()
        except sqlite3.IntegrityError:
            QMessageBox.information(self, "Логин занят", "Введённый логин уже занят!")
        conn.close()



class apanel(QWidget):
        def __init__(self):
            super().__init__()

            self.setWindowTitle("Админ-панель")
            self.setGeometry(300, 300, 450, 410)

            self.table_widget = QTableWidget(self)
            shh_button = QPushButton("Обновить")
            shh_button.clicked.connect(self.ref)
            self.iddell = QLineEdit(self)

            del_button = QPushButton("Удалить")
            del_button.clicked.connect(self.delll)


            self.lexit1 = QPushButton("Выйти")
            self.lexit1.clicked.connect(self.close)
            layout = QVBoxLayout()
            layout.addWidget(shh_button)
            layout.addWidget(del_button)
            layout.addWidget(self.iddell)
            layout.addWidget(self.table_widget)
            layout.addWidget(self.lexit1)

            self.setLayout(layout)

        def delll(self):
            idd = self.iddell.text()
            conn = sqlite3.connect('bb.db')
            cursor = conn.cursor()
            cursor.execute("DELETE FROM users WHERE id = ?", (idd,))
            conn.commit()
            conn.close()

        def ref(self):
                conn = sqlite3.connect('bb.db')
                cursor = conn.cursor()

                cursor.execute("SELECT * FROM users")
                rows = cursor.fetchall()

                self.table_widget.setColumnCount(len(rows[0]))
                self.table_widget.setRowCount(len(rows))
                self.table_widget.setHorizontalHeaderLabels([description[0] for description in cursor.description])

                for i, row in enumerate(rows):
                    for j, cell in enumerate(row):
                        self.table_widget.setItem(i, j, QTableWidgetItem(str(cell)))

                conn.close()

                self.table_widget.resizeColumnsToContents()




if __name__ == '__main__':
    app = QApplication([])
    login_window = LoginWindow()
    login_window.show()
    app.exec_()