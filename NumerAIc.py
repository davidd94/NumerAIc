from flask import Flask, render_template


app = Flask(__name__,
            template_folder='./src',
            static_folder='public',
            static_url_path='')


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8003, debug=True)