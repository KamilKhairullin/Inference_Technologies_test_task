# Тестовое задание Computer Vision Engineer Inference Technologies
Камиль Хайруллин (Kamil Khairullin)

## Как запустить проект.

1. Клонировать реопзиторий 
```
git clone https://github.com/KamilKhairullin/Visual-Transformer.git
```
2.1 Скачать веса для моделей по [ссылке!](http://google.com). Разархивировать и поместить их в папку с репозиторием.
2.2 Скачать датасет fruits-360 и поместить в папку с репозиторием.
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

6.1 Запустить evaluate на претрейнед весах для датасета MNIST
```
exec(open('load_mnist.py').read())
```
6.1 Запустить evaluate на претрейнед весах для датасета Fruits 360
```
exec(open('load_fruits.py').read())
```

