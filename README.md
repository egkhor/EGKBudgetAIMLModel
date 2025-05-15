# EGKBudgetAIMLModel

This repository contains the machine learning model for the EGK BudgetAI iOS app, which uses CoreML to classify transactions and forecast budgets. The model is trained on synthetic transaction data and predicts categories based on amount, day of week, and month.

## Usage
1. Install dependencies: `pip install scikit-learn coremltools pandas numpy`
2. Run `generate_budget_classifier.py` to generate the `BudgetClassifier.mlpackage`.
3. Import the `.mlpackage` into an Xcode project for use with the EGK BudgetAI app.

## License
[MIT License](LICENSE)
