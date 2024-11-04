from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
import json
import os
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io
import base64
import seaborn as sns


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # For session management

EXPENSES_FILE = 'expenses.json'

def load_expenses():
    if os.path.exists(EXPENSES_FILE):
        try:
            with open(EXPENSES_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            # If JSON is invalid, log it and return an empty list
            print("Invalid JSON in expenses.json. Resetting file.")
            return []
    return []


def save_expenses(expenses):
    with open(EXPENSES_FILE, 'w') as f:
        json.dump(expenses, f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_expense', methods=['POST'])
def add_expense():
    amount = request.form['amount']
    category = request.form['category']
    date = request.form['date']
    description = request.form['description']

    expenses = load_expenses()
    expense = {
        'amount': float(amount),
        'category': category,
        'date': date,
        'description': description
    }
    expenses.append(expense)
    save_expenses(expenses)

    flash('Expense added successfully!')
    return redirect(url_for('index'))

@app.route('/view_expenses')
def view_expenses():
    expenses = load_expenses()
    return render_template('view_expenses.html', expenses=expenses)

@app.route('/delete_expense/<int:index>', methods=['POST'])
def delete_expense(index):
    expenses = load_expenses()
    if 0 <= index < len(expenses):
        expenses.pop(index)
        save_expenses(expenses)
        flash('Expense deleted successfully!')
    else:
        flash('Invalid index. Could not delete expense.')
    return redirect(url_for('view_expenses'))


@app.route('/predict_expenses')
def predict_expenses():
    expenses = load_expenses()
    if not expenses:
        return "No expenses to predict."

    df = pd.DataFrame(expenses)
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_year'] = df['date'].dt.dayofyear

    # Prepare data for modeling
    X = df[['day_of_year']]
    y = df['amount']

    model = LinearRegression()
    model.fit(X, y)

    # Predict for the next month
    future_days = np.array([df['day_of_year'].max() + i for i in range(1, 31)]).reshape(-1, 1)
    predictions = model.predict(future_days)

    # Create a line plot for predictions
    plt.figure(figsize=(10, 6))
    plt.plot(future_days, predictions, color='orange', marker='o', label='Predicted Expenses')
    plt.axhline(y=y.mean(), color='r', linestyle='--', label='Average Expense')
    plt.title('Predicted Expenses for the Next Month', fontsize=16, weight='bold')
    plt.xlabel('Day of Year', fontsize=12)
    plt.ylabel('Predicted Amount ($)', fontsize=12)
    plt.legend()
    
    pred_img = io.BytesIO()
    plt.savefig(pred_img, format='png')
    pred_img.seek(0)
    pred_base64 = base64.b64encode(pred_img.getvalue()).decode('utf8')

    return render_template('predict_expenses.html', pred_img=pred_base64)


@app.route('/visualize_expenses')
def visualize_expenses():
    expenses = load_expenses()
    if not expenses:
        return "No expenses to visualize."

    df = pd.DataFrame(expenses)

    # Pie chart for category distribution
    category_counts = df['category'].value_counts()
    plt.figure(figsize=(10, 6))
    plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    plt.title('Expense Distribution by Category', fontsize=16, weight='bold')
    
    pie_img = io.BytesIO()
    plt.savefig(pie_img, format='png')
    pie_img.seek(0)
    pie_base64 = base64.b64encode(pie_img.getvalue()).decode('utf8')
    plt.clf()

    # Bar chart for total expenses per category
    plt.figure(figsize=(10, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values, palette='viridis')
    plt.title('Total Expenses per Category', fontsize=16, weight='bold')
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Total Amount ($)', fontsize=12)

    bar_img = io.BytesIO()
    plt.savefig(bar_img, format='png')
    bar_img.seek(0)
    bar_base64 = base64.b64encode(bar_img.getvalue()).decode('utf8')

    return render_template('visualize_expenses.html', pie_img=pie_base64, bar_img=bar_base64)


if __name__ == '__main__':
    app.run(debug=True)