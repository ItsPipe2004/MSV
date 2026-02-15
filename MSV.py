import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ==========================================
# 1. LÓGICA DE IA (Backend)
# ==========================================

class MotorTerreno:
    def __init__(self):
        
        np.random.seed(42)
        datos = []

        # Generación de datos simulados
        for _ in range(40):
            # Plano (0)
            datos.append([np.random.uniform(2,4),
                          np.random.uniform(3,6),
                          np.random.uniform(10,25),
                          0])
            
            # Fangoso (1)
            datos.append([np.random.uniform(4,6),
                          np.random.uniform(8,12),
                          np.random.uniform(40,60),
                          1])
            
            # Rocoso (2)
            datos.append([np.random.uniform(6,9),
                          np.random.uniform(14,20),
                          np.random.uniform(5,15),
                          2])
            
            # Arenoso (3)
            datos.append([np.random.uniform(3,6),
                          np.random.uniform(6,12),
                          np.random.uniform(20,40),
                          3])

        self.df = pd.DataFrame(datos, columns=["Vibracion", "Pendiente", "Humedad", "Terreno"])

        self.scaler = StandardScaler()
        self.modelo = SVC(kernel='rbf', C=10, probability=True)

        self._entrenar_modelo()

    def _entrenar_modelo(self):
        X = self.df[["Vibracion", "Pendiente", "Humedad"]]
        y = self.df["Terreno"]

        X_scaled = self.scaler.fit_transform(X)
        self.modelo.fit(X_scaled, y)

    def predecir(self, vibracion, pendiente, humedad):
        try:
            nuevo = self.scaler.transform([[vibracion, pendiente, humedad]])
            pred = self.modelo.predict(nuevo)[0]
            prob = max(self.modelo.predict_proba(nuevo)[0]) * 100

            clases = {
                0: "Plano",
                1: "Fangoso",
                2: "Rocoso",
                3: "Arenoso"
            }

            return pred, clases[pred], prob
        except:
            return None, None, 0

# ==========================================
# 2. INTERFAZ GRÁFICA (Frontend)
# ==========================================

class AplicacionGUI:
    def __init__(self, root):
        self.motor = MotorTerreno()
        self.root = root
        self.root.title("Clasificador de Terreno con SVM")
        self.root.geometry("500x420")
        self.root.configure(bg="#f0f0f0")

        style = ttk.Style()
        style.configure("TLabel", font=("Helvetica", 11), background="#f0f0f0")

        # --- Título ---
        lbl_titulo = tk.Label(root, text="Sistema Inteligente de Terreno",
                              font=("Arial", 16, "bold"), bg="#f0f0f0", fg="#333")
        lbl_titulo.pack(pady=20)

        frame_form = tk.Frame(root, bg="#f0f0f0")
        frame_form.pack(pady=10)

        # --- Vibración ---
        ttk.Label(frame_form, text="Vibración (Hz):").grid(row=0, column=0, padx=10, pady=5)
        self.entry_vib = ttk.Entry(frame_form, width=20)
        self.entry_vib.grid(row=0, column=1)

        # --- Pendiente ---
        ttk.Label(frame_form, text="Pendiente (%):").grid(row=1, column=0, padx=10, pady=5)
        self.entry_pen = ttk.Entry(frame_form, width=20)
        self.entry_pen.grid(row=1, column=1)

        # --- Humedad ---
        ttk.Label(frame_form, text="Humedad (%):").grid(row=2, column=0, padx=10, pady=5)
        self.entry_hum = ttk.Entry(frame_form, width=20)
        self.entry_hum.grid(row=2, column=1)

        # --- Botón ---
        btn = tk.Button(root, text="CLASIFICAR TERRENO",
                        bg="#007bff", fg="white",
                        font=("Arial", 10, "bold"),
                        command=self.mostrar_prediccion,
                        padx=10, pady=5, relief="flat")
        btn.pack(pady=20)

        # --- Resultado ---
        self.lbl_resultado = tk.Label(root,
                                      text="Esperando datos...",
                                      font=("Arial", 12),
                                      bg="#e0e0e0",
                                      width=45,
                                      height=4)
        self.lbl_resultado.pack(pady=10)

    def mostrar_prediccion(self):
        try:
            vib = float(self.entry_vib.get())
            pen = float(self.entry_pen.get())
            hum = float(self.entry_hum.get())

            pred_idx, pred_texto, confianza = self.motor.predecir(vib, pen, hum)

            if pred_texto:

                if pred_idx == 0:
                    navegabilidad = "✅ Adecuado para navegación autónoma"
                    color = "#28a745"
                else:
                    navegabilidad = "⚠ No recomendado para navegación autónoma"
                    color = "#dc3545"

                texto = f"Terreno detectado: {pred_texto}\nConfianza del modelo: {confianza:.1f}%\n{navegabilidad}"
                self.lbl_resultado.config(text=texto, fg=color, bg="#fff")

                self.graficar(vib, pen)

            else:
                messagebox.showerror("Error", "No se pudo realizar la predicción")

        except Exception as e:
            messagebox.showerror("Error", str(e))


    def graficar(self, vib, pen):

        plt.figure(figsize=(8,6))

        nombres = {
            0: "Plano",
            1: "Fangoso",
            2: "Rocoso",
            3: "Arenoso"
        }

        colores = {
            0: "#2ecc71",
            1: "#e74c3c",
            2: "#7f8c8d",
            3: "#f1c40f"
        }

        for i in range(4):
            subset = self.motor.df[self.motor.df["Terreno"] == i]
            plt.scatter(
                subset["Vibracion"],
                subset["Pendiente"],
                color=colores[i],
                label=nombres[i],
                alpha=0.7,
                edgecolors='black'
            )

        plt.scatter(
            vib,
            pen,
            color="black",
            s=250,
            marker="X",
            edgecolors="white",
            linewidth=2,
            label="Nuevo dato"
        )

        plt.title("Clasificación de Terreno mediante SVM", fontsize=14, fontweight="bold")
        plt.xlabel("Vibración (Hz)", fontsize=12)
        plt.ylabel("Pendiente (%)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    root = tk.Tk()
    app = AplicacionGUI(root)
    root.mainloop()