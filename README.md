# Тестовое задание Computer Vision Engineer Inference Technologies
Камиль Хайруллин (Kamil Khairullin)

## Как запустить проект.

1. Клонировать реопзиторий 
```
git clone https://github.com/KamilKhairullin/Visual-Transformer.git
```
2.1 Скачать веса для моделей по [ссылке](https://drive.google.com/file/d/1buYfAOxozvR_zi-Yyn-KmPhgzMJumuUl/view?usp=sharing). Разархивировать и поместить их в папку с репозиторием. <br/>
2.2 Скачать датасет [fruits 360](https://www.kaggle.com/moltean/fruits) и поместить в папку с репозиторием.
После скачивания папка должна выглядеть так <br/>
![Screenshot 2021-05-07 at 22 27 00](https://user-images.githubusercontent.com/54369751/117499113-6da46900-af83-11eb-926a-33379e22a774.png)
 <br/>
3. Открыть консоль в папке с репозиторием и забилдить docker image
```
docker build -t name .
```

4. Выделите docker достаточно ресурсов. Мои настройки docker
![Screenshot 2021-05-07 at 22 30 53](https://user-images.githubusercontent.com/54369751/117499423-e3a8d000-af83-11eb-9ce3-b8275bda68da.png)

5. Запустить docker container
```
docker run  -it name
```
<br/>
После запуска контейнера должен появится Python SHELL <br/>

![Screenshot 2021-05-07 at 22 41 47](https://user-images.githubusercontent.com/54369751/117500513-6bdba500-af85-11eb-9def-0b2aceb99b7f.png)

<br/>

6.1 Запустить evaluate на претрейнед весах для датасета MNIST
```
exec(open('load_mnist.py').read())
```
6.2 Запустить evaluate на претрейнед весах для датасета Fruits 360
```
exec(open('load_fruits.py').read())
```
6.3 Запустить training для датасета MNIST
```
exec(open('train_mnist.py').read())
```
6.4 Запустить training для датасета Fruits 360
```
exec(open('train_fruits.py').read())
```
<br/>


References:

1. [Keras Image Classification with Vision Transformer](https://keras.io/examples/vision/image_classification_with_vision_transformer/)
