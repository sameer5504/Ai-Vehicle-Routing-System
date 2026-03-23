# 🚚 Delivery Route Optimizer (GA + SA)

An AI-based delivery optimization system that solves a Vehicle Routing Problem (VRP) using two metaheuristic algorithms:

- Genetic Algorithm (GA)
- Simulated Annealing (SA)

The system includes an interactive graphical user interface (GUI) to visualize routes, compare algorithms, and analyze convergence performance.

---

## 📌 Features

- 🧠 Dual optimization algorithms (GA & SA)
- ⚖️ Compare mode to select the best solution automatically
- 🚗 Multi-vehicle delivery with capacity constraints
- 📦 Package priority handling
- 📊 Convergence visualization (algorithm performance over time)
- 🗺️ Route plotting with Matplotlib
- 🖥️ GUI interface built with Tkinter
- 📁 Export results to file
- ✅ Input validation

---

## 📌 Problem Description

This project solves a simplified Vehicle Routing Problem:

- A set of delivery packages with:
  - Location (x, y)
  - Weight
  - Priority
- A fleet of vehicles with limited capacity

Goal:
Minimize total delivery cost (distance + penalties) while respecting constraints.

---

## 📌 Algorithms Used

### 🔹 Genetic Algorithm (GA)
- Population-based optimization
- Crossover & mutation
- Evolves both:
  - Vehicle assignment
  - Delivery order

### 🔹 Simulated Annealing (SA)
- Probabilistic local search
- Accepts worse solutions early (exploration)
- Gradually converges to optimal solutions

---

## 📌 Input Format

Create a `.txt` file:
cooling_rate
vehicle_capacities
x y weight priority
x y weight priority
...

### Example:
0.95
15 12 10
2 3 4 5
5 7 3 2
8 2 6 4

---

## 📌 How to Run

### 1. Install dependencies

pip install matplotlib

### 2. Run the program

### 3. Use the GUI
- Load input file
- Select algorithm:
  - GA
  - SA
  - Compare
- Click **Run Optimization**

---

## 📌 Output

- Best solution cost
- Vehicle routes
- Route visualization
- Convergence graph
- Execution logs

---

## 📌 Project Structure
├── upgraded_delivery_optimizer.py
├── input.txt
├── README.md

---

## 📌 Technologies Used

- Python
- Tkinter (GUI)
- Matplotlib (Visualization)
- NumPy (optional if used)

---

## 📌 Future Improvements

- Real-world map integration (Google Maps / OpenStreetMap)
- Time window constraints
- Multi-objective optimization
- Web-based interface (Flask or React)

---

## 📌 Author

- Samir Ali

---

