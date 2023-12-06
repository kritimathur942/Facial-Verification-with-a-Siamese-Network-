from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator

# Load JSON model structure
json_file = open("emotion_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()

# Load the model architecture
loaded_model = model_from_json(loaded_model_json)

# Load model weights into the architecture
loaded_model.load_weights("emotion_model.h5")
print("Loaded model from disk")

# Reinitialize the data generators for validation
validation_data_gen = ImageDataGenerator(rescale=1.0/255)
validation_generator = validation_data_gen.flow_from_directory(
    'data/test',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

# Evaluate the loaded model on the test data
test_loss, test_accuracy = loaded_model.evaluate(validation_generator)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')
