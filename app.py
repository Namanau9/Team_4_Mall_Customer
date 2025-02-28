from flask import Flask, render_template
from kmeans_plot import generate_kmeans_plot
from regression_plot import generate_regression_plot

app = Flask(__name__)

@app.route('/')
def index():
    generate_kmeans_plot()
    generate_regression_plot()
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
