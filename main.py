from website import create_app
from flask import Flask
app = Flask(__name__, static_url_path='/static')

app = create_app()

if __name__ == '__main__':
    app.run(debug=True) 