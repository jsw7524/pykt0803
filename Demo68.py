import keras.utils as utils

orig = [0, 5, 7, 9, 4,14]
NUM_DIGITS = 15
print(f"before conversion, data={orig}")
converted = utils.to_categorical(orig, NUM_DIGITS)
print(f"after conversion, data={converted}")