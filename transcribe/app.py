from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import csv
from supabase import create_client, Client
from dotenv import load_dotenv
import os


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_API_KEY = os.getenv('SUPABASE_API_KEY')
supabase: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)


@app.route('/')
def index():
    response = supabase.table('transcriptions').select('audio_file_name, transcript').execute()
    transcripts = response.data
    return render_template('index.html', transcripts=transcripts)

@app.route('/show_upload')
def show_upload_action():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload():
    audio_files = request.files.getlist('audio_files')
    for audio_file in audio_files:
        audio_file.save(os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename))
    return redirect(url_for('transcribe'))


@app.route('/transcribe')
def transcribe():
    response = supabase.table('transcriptions').select('audio_file_name, transcript').execute()
    transcripts = {file['audio_file_name']: file['transcript'] for file in response.data if not file['transcript']}
    audio_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f not in transcripts.keys()]
    return render_template('transcribe.html', audio_files=audio_files)



@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/list', methods=['GET', 'POST'])
def list_files():
    if request.method == 'POST':
        updated_transcriptions = request.form.to_dict()
        for audio_file, transcription in updated_transcriptions.items():
            response = supabase.table('transcriptions').update({
                'transcript': transcription
            }).match({'audio_file_name': audio_file}).execute()
    response = supabase.table('transcriptions').select('*').execute()
    files = response.data
    return render_template('list.html', files=files)



@app.route('/save_transcriptions', methods=['POST'])
def save_transcriptions():
    transcriptions = request.form.to_dict()
    with open('transcriptions.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Audio File', 'Transcription'])
        for audio_file, transcription in transcriptions.items():
            response = supabase.table("transcriptions").insert({
                'audio_file_name': audio_file,
                'transcript': transcription,
                 'status': 'complete'
            }).execute()

            csv_writer.writerow([audio_file, transcription])
    return redirect(url_for('index'))

@app.route('/update_transcription', methods=['POST'])
def update_transcription():
    audio_file = request.form['audio_file']
    updated_transcription = request.form['transcription']
    response = supabase.table('transcriptions').update({
        'transcript': updated_transcription
    }).match({'audio_file_name': audio_file}).execute()
    return redirect(url_for('list_files'))

@app.route('/delete_transcription', methods=['POST'])
def delete_transcription():
    audio_file = request.form['audio_file']
    response = supabase.table('transcriptions').delete().match({'audio_file_name': audio_file}).execute()
    return redirect(url_for('list_files'))




if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)

