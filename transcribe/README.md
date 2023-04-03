# Transcription App

A simple web application for manually transcribing audio files. Users can upload audio files, listen to them, and type their transcriptions. The application stores the transcriptions in a Supabase database and allows users to review and rate the quality of the transcriptions.

## Features

- Upload audio files
- Manually transcribe audio files by listening and typing the transcription
- Save transcriptions to a Supabase database
- View and edit transcriptions
- Review and rate transcriptions for quality
- Simple navigation between different pages

## Installation

### Prerequisites

- Python 3.7 or higher
- pip

### Steps

1. Clone the repository:

```
git clone https://github.com/yourusername/transcription-app.git
cd transcription-app
```

2. Create a virtual environment and install the required packages:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Set up your Supabase database and obtain your Supabase URL and API key. You can follow the [Supabase documentation](https://supabase.io/docs/guides/database) for instructions on setting up your database.

4. Create a `.env` file in the project root folder with the following content:

```
SUPABASE_URL=your_supabase_url
SUPABASE_API_KEY=your_supabase_api_key
```

Replace `your_supabase_url` and `your_supabase_api_key` with the appropriate values from your Supabase account.

5. Run the application:

```
python app.py
```

6. Open your web browser and navigate to `http://127.0.0.1:5000` to access the application.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
