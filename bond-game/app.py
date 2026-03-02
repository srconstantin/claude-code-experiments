from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    print("Bond Game running at http://localhost:5050")
    app.run(debug=True, port=5050)
