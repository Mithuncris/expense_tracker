from flask import Flask, render_template, request, redirect, flash
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import base64
import io
import json

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Sample data to simulate expenses (You should load this from your expenses.json)
expenses = []

# Load expenses from JSON file
def load_expenses():
    global expenses
    try:
        with open('expenses.json', 'r') as file:
            expenses = json.load(file)
    except FileNotFoundError:
        expenses = []

# Save expenses to JSON file
def save_expenses():
    with open('expenses.json', 'w') as file:
        json.dump(expenses, file)

# Predict future expenses based on historical data
def predict_expenses():
    df = pd.DataFrame(expenses)
    
    # Feature Engineering
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek

    if df.empty:
        return None  # Not enough data to predict

    X = df[['amount', 'month', 'day_of_week']]
    y = df['amount']  # Assuming you want to predict future amounts

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a model
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    # Predict future expenses (e.g., for the next month)
    future_data = pd.DataFrame({
        'amount': [0] * 30,  # Placeholder for next month's expenses
        'month': [df['month'].max() + 1] * 30,
        'day_of_week': list(range(0, 30))
    })
    
    predictions = model.predict(future_data)
    
    # Plot predictions
    plt.figure(figsize=(10, 5))
    plt.plot(predictions, label='Predicted Expenses', color='blue')
    plt.title('Predicted Expenses for Next Month')
    plt.xlabel('Days')
    plt.ylabel('Predicted Amount')
    plt.legend()
    
    # Save to a BytesIO stream
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    pred_img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return pred_img

@app.route('/')
def index():
    load_expenses()
    return render_template('index.html')

@app.route('/add_expense', methods=['POST'])
def add_expense():
    amount = request.form['amount']
    category = request.form['category']
    date = request.form['date']
    description = request.form['description']
    
    expenses.append({
        'amount': float(amount),
        'category': category,
        'date': date,
        'description': description
    })
    
    save_expenses()
    flash('Expense added successfully!')
    return redirect('/')

@app.route('/view_expenses')
def view_expenses():
    load_expenses()
    return render_template('view_expenses.html', expenses=expenses)

@app.route('/delete_expense/<int:index>', methods=['POST'])
def delete_expense(index):
    if 0 <= index < len(expenses):
        del expenses[index]
        save_expenses()
        flash('Expense deleted successfully!')
    else:
        flash('Error: Expense not found.')
    return redirect('/view_expenses')

@app.route('/predict_expenses')
def predict_expenses_view():
    pred_img = predict_expenses()
    return render_template('predict_expenses.html', pred_img=pred_img)

@app.route('/visualize_expenses')
def visualize_expenses():
    load_expenses()
    df = pd.DataFrame(expenses)
    category_totals = df.groupby('category')['amount'].sum()

    # Create Pie Chart
    plt.figure(figsize=(8, 8))
    category_totals.plot.pie(autopct='%1.1f%%')
    plt.title('Expense Distribution by Category')
    
    buf_pie = io.BytesIO()
    plt.savefig(buf_pie, format='png')
    buf_pie.seek(0)
    pie_img = base64.b64encode(buf_pie.read()).decode('utf-8')
    plt.close()

    # Create Bar Chart
    plt.figure(figsize=(10, 5))
    category_totals.plot.bar()
    plt.title('Total Expenses per Category')
    plt.xlabel('Category')
    plt.ylabel('Total Amount')

    buf_bar = io.BytesIO()
    plt.savefig(buf_bar, format='png')
    buf_bar.seek(0)
    bar_img = base64.b64encode(buf_bar.read()).decode('utf-8')
    plt.close()

    return render_template('visualize_expenses.html', pie_img=pie_img, bar_img=bar_img)

if __name__ == '__main__':
    app.run(debug=True)
