import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Krok 1: Przygotowanie danych
train_dir = 'data/train'  # Ścieżka do folderu z danymi treningowymi
test_dir = 'data/test1'   # Ścieżka do folderu z danymi testowymi

# Tworzenie generatorów danych z augmentacją
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Wczytywanie danych z podziałem na klasy
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'  # Zmiana na 'categorical' dla klasyfikacji wieloklasowej
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'  # Zmiana na 'categorical' dla klasyfikacji wieloklasowej
)

# Krok 2: Tworzenie modelu sieci neuronowej
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=len(train_generator.class_indices), activation='softmax'))  # Zmiana na 'softmax'

# Kompilacja modelu
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Zmiana na 'categorical_crossentropy'

# Krok 3: Trenowanie modelu
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=test_generator,
    validation_steps=len(test_generator)
)

# Krok 4: Wyświetlanie wyników trenowania
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Dokładność na przestrzeni epok')
plt.xlabel('Epoki')
plt.ylabel('Dokładność')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Straty na przestrzeni epok')
plt.xlabel('Epoki')
plt.ylabel('Strata')
plt.legend()
plt.show()

# Krok 5: Ocena modelu na danych testowych
loss, accuracy = model.evaluate(test_generator)
print(f'Accuracy on test data: {accuracy*100:.2f}%')

# Sprawdzenie liczby obrazów w generatorze treningowym
print(f'Liczba obrazów w zbiorze treningowym: {train_generator.samples}')
print(f'Liczba klas w zbiorze treningowym: {len(train_generator.class_indices)}')

# Sprawdzenie liczby obrazów w generatorze testowym
print(f'Liczba obrazów w zbiorze testowym: {test_generator.samples}')
print(f'Liczba klas w zbiorze testowym: {len(test_generator.class_indices)}')

# Wyświetlanie mapy klas
print("Klasy:", train_generator.class_indices)
