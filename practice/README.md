# Практика по OpenVINO

Запись практики №1 - Основы работы с библиотекой OpenCV:
https://vk.com/videos-2139021?z=video-2139021_456239075

# Часто возникающие проблемы

## Проблема с импортом cv2
```bash
Traceback (most recent call last):
  File "ie_detector.py", line 5, in <module>
    import cv2
ImportError: DLL load failed while importing cv2: Не найден указанный модуль.
```

В запускаемом python файле импортируйте вручную пути к папкам с динамическими библиотеками 
```python
import os
os.add_dll_directory("C:\\Program Files (x86)\\Intel\\openvino_2021.3.394\\deployment_tools\\ngraph\\lib")
os.add_dll_directory("C:\\Program Files (x86)\\Intel\\openvino_2021.3.394\\deployment_tools\\inference_engine\\external\\tbb\\bin")
os.add_dll_directory("C:\\Program Files (x86)\\Intel\\openvino_2021.3.394\\deployment_tools\\inference_engine\\bin\\intel64\\Release")
#os.add_dll_directory("C:\\Program Files (x86)\\Intel\\openvino_2021.3.394\\deployment_tools\\inference_engine\\external\\hddl\\bin")
os.add_dll_directory("C:\\Program Files (x86)\\Intel\\openvino_2021.3.394\\opencv\\bin")
```


## Не удается найти файл libs

```bash
Traceback (most recent call last):
  File "ie_detector.py", line 13, in <module>
    from openvino.inference_engine import IECore
  File "C:\Program Files (x86)\Intel\openvino_2021.3.394\python\python3.8\openvino\inference_engine\__init__.py", line 30, in <module>
    os.add_dll_directory(os.path.abspath(openvino_dlls))
  File "C:\Program Files\Python38\lib\os.py", line 1109, in add_dll_directory
    cookie = nt._add_dll_directory(path)
FileNotFoundError: [WinError 2] Не удается найти указанный файл: 'C:\\Program Files (x86)\\Intel\\openvino_2021.3.394\\python\\python3.8\\openvino\\libs'
```

Создайте пустую папку "C:\\Program Files (x86)\\Intel\\openvino_2021.3.394\\python\\python3.8\\openvino\\libs"
