from flask import Flask, render_template
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route('/home')
@cross_origin()
def home():
  return render_template('index.html')

app.run(host="0.0.0.0", port=5000, debug=True)
