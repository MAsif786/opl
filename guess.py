from sklearn.tree import DecisionTreeClassifier
import numpy as np

class NumberPredictor:
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.model = DecisionTreeClassifier()
        self.X = []
        self.y = []
        self.trained = False

    def add_sequence(self, sequence):
        # build dataset using sliding window
        for i in range(len(sequence) - self.window_size):
            x = sequence[i:i+self.window_size]
            y = sequence[i+self.window_size]
            self.X.append(x)
            self.y.append(y)

    def train(self):
        if len(self.X) == 0:
            print("Not enough data to train.")
            return

        self.model.fit(self.X, self.y)
        self.trained = True

    def predict_next(self, recent_numbers):
        if not self.trained:
            return None

        if len(recent_numbers) < self.window_size:
            return None

        x = np.array(recent_numbers[-self.window_size:]).reshape(1, -1)
        return self.model.predict(x)[0]


# ------------------- DEMO -------------------

data = []

print("Enter numbers one by one. Type 'q' to quit.\n")

predictor = NumberPredictor(window_size=3)

while True:
    val = input("Enter number: ")

    if val.lower() == 'q':
        break

    try:
        num = int(val)
    except ValueError:
        print("Enter valid integer.")
        continue

    data.append(num)

    # retrain every time we get new data (simple approach)
    predictor = NumberPredictor(window_size=3)
    predictor.add_sequence(data)
    predictor.train()

    guess = predictor.predict_next(data)

    print("Current sequence:", data)
    print("ML guess for next number:", guess)
    print()
