import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import ttkbootstrap as ttk
import tkinter as tk
from ttkbootstrap.constants import *
import numpy as np



# ========================= LOAD DATA ==========================
data = pd.read_csv(r'C:\Users\win10\OneDrive\Desktop\AIS\coding\cooding2\Real\real_estate_data.csv')

x = data.drop(columns=['No', 'Y house price of unit area'])
y = data['Y house price of unit area']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    random_state=42
)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


# ========================= MAIN WINDOW ==========================
app = ttk.Window(themename="superhero")
app.title("🏡 Real Estate Price Predictor")
app.geometry("820x750")
app.resizable(False, False)


# ========================= POPUP FUNCTION ==========================
def show_popup(title, message, color):
    popup = ttk.Toplevel(app)
    popup.title(title)
    popup.geometry("450x300")
    ttk.Label(
        popup, text=message, bootstyle=color,
        font=("Helvetica", 13), wraplength=400
    ).pack(pady=20)
    ttk.Button(
        popup, text="Close", command=popup.destroy,
        bootstyle=SECONDARY
    ).pack(pady=10)


# ========================= PREDICT FUNCTION ==========================
def predict_price():
    try:
        new_land = pd.DataFrame({
            'X1 transaction date': [float(entry_date.get())],
            'X2 house age': [float(entry_age.get())],
            'X3 distance to the nearest MRT station': [float(entry_distance.get())],
            'X4 number of convenience stores': [int(entry_stores.get())],
            'X5 latitude': [float(entry_latitude.get())],
            'X6 longitude': [float(entry_longitude.get())]
        })

        predicted_price = model.predict(new_land)[0]

        show_popup(
            "Prediction Result",
            f"💰 Predicted Price: ${predicted_price:,.2f}\n\n"
            f"📉 MAE: ±${mae:,.2f}\n"
            f"📉 RMSE: ±${rmse:,.2f}\n"
            f"📈 R² Score: {r2:.2f}",
            INFO
        )
    except:
        show_popup("Input Error", "Please enter valid numbers in all fields!", DANGER)


# ========================= RESET FUNCTION ==========================
def reset_fields():
    for entry in entries.values():
        entry.delete(0, tk.END)


# ========================= MODEL INFO ==========================
def show_model_info():
    show_popup(
        "Model Information",
        f"📌 Model: Random Forest Regressor\n"
        f"📊 Training samples: {len(x_train)}\n"
        f"📊 Test samples: {len(x_test)}\n\n"
        f"MAE: {mae:.2f}\n"
        f"RMSE: {rmse:.2f}\n"
        f"R² Score: {r2:.2f}",
        PRIMARY
    )


# ========================= TITLE ==========================
title_label = ttk.Label(
    app, text="🏡 Real Estate Price Predictor",
    font=("Helvetica", 24, "bold"),
    bootstyle=INFO
)
title_label.pack(pady=20)


# ========================= FIELDS FRAME ==========================
card = ttk.Frame(app, padding=20, borderwidth=2, relief="ridge")
card.pack(pady=10)

fields = [
    ("Transaction Date:", "entry_date"),
    ("House Age:", "entry_age"),
    ("Distance to MRT (m):", "entry_distance"),
    ("Number of Stores:", "entry_stores"),
    ("Latitude:", "entry_latitude"),
    ("Longitude:", "entry_longitude"),
]

entries = {}

for i, (label_text, key) in enumerate(fields):
    ttk.Label(card, text=label_text, font=("Helvetica", 13)).grid(
        row=i, column=0, pady=10, padx=10, sticky=W
    )
    entry = ttk.Entry(card, font=("Helvetica", 13), width=25)
    entry.grid(row=i, column=1, pady=10, padx=10)
    entries[key] = entry

entry_date = entries["entry_date"]
entry_age = entries["entry_age"]
entry_distance = entries["entry_distance"]
entry_stores = entries["entry_stores"]
entry_latitude = entries["entry_latitude"]
entry_longitude = entries["entry_longitude"]


# ========================= BUTTONS ==========================
btn_frame = ttk.Frame(app)
btn_frame.pack(pady=25)

ttk.Button(btn_frame, text="Predict Price", command=predict_price,
           width=20, bootstyle=SUCCESS).grid(row=0, column=0, padx=10)

ttk.Button(btn_frame, text="Reset", command=reset_fields,
           width=20, bootstyle=WARNING).grid(row=0, column=1, padx=10)

ttk.Button(btn_frame, text="Model Info", command=show_model_info,
           width=20, bootstyle=INFO).grid(row=0, column=2, padx=10)

ttk.Button(app, text="Exit", command=app.destroy,
           width=20, bootstyle=DANGER).pack(pady=10)


# ========================= RUN ==========================
app.mainloop()