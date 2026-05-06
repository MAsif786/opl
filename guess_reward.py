from collections import defaultdict
import random

class PenaltyLearner:
    def __init__(self):
        # transition table: {current_number: {next_number: weight}}
        self.transitions = defaultdict(lambda: defaultdict(float))

    def update(self, sequence):
        """
        Learn patterns from full sequence
        """
        for i in range(len(sequence) - 1):
            a, b = sequence[i], sequence[i + 1]
            self.transitions[a][b] += 1.0  # reward observed transition

    def predict(self, last_num):
        """
        Predict next number based on weighted probabilities
        """
        options = self.transitions.get(last_num)

        if not options:
            return None

        total = sum(options.values())
        choices = list(options.keys())
        weights = [options[c] / total for c in choices]

        return random.choices(choices, weights=weights)[0]

    def penalize(self, last_num, wrong_guess):
        """
        Penalize wrong predictions
        """
        if wrong_guess in self.transitions[last_num]:
            self.transitions[last_num][wrong_guess] *= 0.7  # reduce confidence


def main():
    data = []
    model = PenaltyLearner()

    print("Penalty-Based Number Predictor")
    print("Enter numbers one by one (type 'q' to quit)\n")

    while True:
        user_input = input("Enter number: ")

        if user_input.lower() == 'q':
            break

        try:
            num = int(user_input)
        except ValueError:
            print("Invalid input. Enter an integer.\n")
            continue

        # make prediction BEFORE updating model
        if data:
            last = data[-1]
            guess = model.predict(last)

            print(f"Prediction: {guess}")

            # apply penalty if wrong
            if guess is not None and guess != num:
                model.penalize(last, guess)

        data.append(num)
        model.update(data)

        print("Current data:", data)
        print("Transitions learned:", dict(model.transitions))
        print("-" * 40)


if __name__ == "__main__":
    main()
