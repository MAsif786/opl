import tkinter as tk
import random
from collections import defaultdict

class QLearningPredictor:
    def __init__(self, alpha=0.3, gamma=0.9):
        self.q = defaultdict(lambda: defaultdict(float))
        self.alpha = alpha
        self.gamma = gamma

    def predict(self, state):
        if state not in self.q:
            return random.randint(0, 9)

        return max(self.q[state], key=self.q[state].get)

    def update(self, state, action, reward, next_state):
        max_next = max(self.q[next_state].values(), default=0)

        old_value = self.q[state][action]
        self.q[state][action] = old_value + self.alpha * (
            reward + self.gamma * max_next - old_value
        )


class GuessGameGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎮 Number Pattern RL Game")

        self.model = QLearningPredictor()

        self.last_number = None
        self.correct = 0
        self.total = 0

        # --- UI ---
        self.label = tk.Label(root, text="Enter a number (0-9):", font=("Arial", 14))
        self.label.pack(pady=10)

        self.entry = tk.Entry(root, font=("Arial", 14))
        self.entry.pack()

        self.button = tk.Button(root, text="Submit", command=self.submit)
        self.button.pack(pady=10)

        self.pred_label = tk.Label(root, text="Prediction: -", font=("Arial", 12))
        self.pred_label.pack(pady=5)

        self.acc_label = tk.Label(root, text="Accuracy: 0%", font=("Arial", 12))
        self.acc_label.pack(pady=5)

        self.history = tk.Text(root, height=10, width=40)
        self.history.pack(pady=10)

    def submit(self):
        val = self.entry.get()

        if not val.isdigit():
            return

        num = int(val)

        # prediction step
        if self.last_number is not None:
            prediction = self.model.predict(self.last_number)
            self.pred_label.config(text=f"Prediction: {prediction}")

            reward = 1 if prediction == num else -1

            if prediction == num:
                self.correct += 1

            self.total += 1

            # RL update
            self.model.update(self.last_number, prediction, reward, num)

            # accuracy update
            accuracy = (self.correct / self.total) * 100
            self.acc_label.config(text=f"Accuracy: {accuracy:.2f}%")

        # log
        self.history.insert(tk.END, f"Input: {num}\n")

        self.last_number = num
        self.entry.delete(0, tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    game = GuessGameGUI(root)
    root.mainloop()
