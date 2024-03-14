import cv2
import numpy as np

def find_paper_corners(image):
    # Преобразование изображения в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Детекция углов с помощью алгоритма Харриса
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)

    # Улучшение детекции углов с помощью дилатации
    corners = cv2.dilate(corners, None)

    # Задание порогового значения для отбора сильных углов
    threshold = 0.01 * corners.max()

    # Фильтрация и сохранение координат сильных углов
    corner_coordinates = np.where(corners > threshold)
    corner_coordinates = np.column_stack((corner_coordinates[1], corner_coordinates[0]))  # Изменение формата координат

    # Применение алгоритма RANSAC для поиска четырех углов листа бумаги
    _, mask = cv2.findHomography(corner_coordinates.astype(np.float32),
                                 np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float32),
                                 cv2.RANSAC)

    # Преобразование координат углов с помощью найденной гомографии
    transformed_corners = cv2.perspectiveTransform(corner_coordinates.reshape(-1, 1, 2), mask)

    # Извлечение координат четырех углов
    paper_corners = transformed_corners.squeeze().astype(int)

    return paper_corners

def stretch_paper(image, paper_corners):
    # Определение размеров исходного изображения
    height, width = image.shape[:2]

    # Определение координат четырех углов исходного листа бумаги
    paper_top_left, paper_top_right, paper_bottom_right, paper_bottom_left = paper_corners

    # Определение координат углов нового листа бумаги (всего изображения)
    new_top_left = (0, 0)
    new_top_right = (width - 1, 0)
    new_bottom_right = (width - 1, height - 1)
    new_bottom_left = (0, height - 1)

    # Создание матрицы преобразования перспективы
    matrix = cv2.getPerspectiveTransform(np.float32([paper_top_left, paper_top_right, paper_bottom_right, paper_bottom_left]),
                                         np.float32([new_top_left, new_top_right, new_bottom_right, new_bottom_left]))

    # Применение преобразования перспективы для растягивания листа бумаги на всё изображение
    stretched_image = cv2.warpPerspective(image, matrix, (width, height))

    return stretched_image

# Загрузка изображения
image = cv2.imread('C:\\Users\\Papech\\Desktop\\baza.jpg')

# Нахождение координат углов листа бумаги
paper_corners = find_paper_corners(image)

# Растягивание листа бумаги на всё изображение
stretched_image = stretch_paper(image, paper_corners)

# Вывод результата
cv2.imshow('Stretched Image', stretched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
