from flask import Flask, flash, request, redirect
from flask import render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import rag
import os, json
from apiflask import APIFlask

load_dotenv()

app = APIFlask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'json'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def main():
    return render_template("upload.html")


@app.route("/query", methods=["GET"])
def ask():
    try:
        question = request.args.get("question", "")
        if question == '':
            return redirect('question.html')
        else:
            user_query = question
            answer = rag.get_answer(user_query)
            json_answer = json.dumps({'answer': answer})
            return render_template('answer.html', question=user_query, answer=answer)
    except Exception as e:
        return str(e)


@app.route("/question")
def question():
    return render_template('question.html')


@app.route('/upload', methods=['GET', 'POST'])
async def upload_file():
    if request.method == 'GET':
        return render_template('upload.html')
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        print("Uploaded file is:", file.filename)
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file.content_type != 'application/json':
            flash('Only json files are allowed')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file.seek(0)
            data = json.load(file)
            await rag.create_embeddings(data)
        return render_template("acknowledgement.html", name=file.filename)
    else:
        return render_template("upload.html")

if __name__ == '__main__':
    app.run(debug=True)