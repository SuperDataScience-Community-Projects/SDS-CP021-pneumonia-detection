# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 10:13:38 2025

@author: Marcin Krawczyk
"""

import cv2

# Wczytanie obrazu
image = cv2.imread('../../data/test/NORMAL/IM-0001-0001.jpeg')  # Zastąp 'path_to_image.jpg' ścieżką do Twojego obrazu

# Wyświetlenie obrazu
cv2.imshow('Obraz', image)

# Oczekiwanie na dowolny klawisz
cv2.waitKey(0)  # 0 oznacza, że program czeka na dowolny klawisz, aby zamknąć okno

# Zamknięcie wszystkich okien
cv2.destroyAllWindows()

