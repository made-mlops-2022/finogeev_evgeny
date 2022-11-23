# HW2 MADE

## Сборка докера

`docker build -t inference .`

`docker run -it --rm -p 8000:8000 inference`

## Запуск контейнера 

Можно указать параметр `MODEL_URL` - ссылка на модель в yandex disk.

Решающее дерево (default)

`docker run -it --rm -p 8000:8000 inference`

Случайны лес

`docker run -it --rm -p 8000:8000 -e MODEL_URL=https://disk.yandex.com/d/hoMM56W_p4PZsA inference`

## Запросы в сервис

Решающее дерево (default)

`bash request_script_tree.sh`

Случайны лес

`bash request_script_forest.sh`

