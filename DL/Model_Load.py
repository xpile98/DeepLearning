from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)     # data => [train / test]
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)               # data => [(train / valid) / test]

# >> scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# >> make model
model = keras.models.load_model('Keras_WideDeep3.h5')

# >> model compile and fit
model.compile(loss=['mse', 'mse'], loss_weights=[0.9, 0.1], optimizer=keras.optimizers.SGD(lr=1e-3))
X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]   # input_A : 모든 데이터에서 0,1,2,3,4 index 5개 값만 쓰겠다.
X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]   # input_B : 모든 데이터에서 2,3,4,5,6,7 index 6개 값만 쓰겠다.
X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]

history = model.fit(
    [X_train_A, X_train_B], [y_train, y_train], epochs=20,
    validation_data=([X_valid_A, X_valid_B], [y_valid, y_valid]))
total_loss, main_loss, aux_loss = model.evaluate((X_test_A, X_test_B), (y_test, y_test))
y_pred_main, y_pred_aux = model.predict((X_new_A, X_new_B))
print(y_pred_main, y_pred_aux)
keras.utils.plot_model(model, show_shapes=True, to_file='Keras_WideDeep3.png')

# >> learning curve
import pandas as pd
import matplotlib.pyplot as plt
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.show()

