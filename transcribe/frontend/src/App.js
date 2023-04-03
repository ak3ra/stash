import React, { useState, useEffect } from "react";
import axios from "axios";

function App() {
  const [transcripts, setTranscripts] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    async function fetchTranscripts() {
      const response = await axios.get("/api/transcripts");
      setTranscripts(response.data);
    }
    fetchTranscripts();
  }, []);

  const navigate = (step) => {
    const newIndex = currentIndex + step;
    if (newIndex >= 0 && newIndex < transcripts.length) {
      setCurrentIndex(newIndex);
    }
  };

  const handleThumbsUp = () => {
    const transcript = transcripts[currentIndex];
    axios.post(`/api/transcripts/${transcript.id}/feedback`, {
      feedback: "thumbs_up",
    });
  };

  const handleThumbsDown = () => {
    const transcript = transcripts[currentIndex];
    axios.post(`/api/transcripts/${transcript.id}/feedback`, {
      feedback: "thumbs_down",
    });
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    const transcript = transcripts[currentIndex];
    const updatedTranscript = {
      ...transcript,
      transcript: event.target.elements.transcript.value,
    };
    const response = await axios.post(
      `/api/transcripts/${transcript.id}`,
      updatedTranscript
    );
    const updatedTranscripts = transcripts.slice();
    updatedTranscripts[currentIndex] = response.data;
    setTranscripts(updatedTranscripts);
  };

  if (transcripts.length === 0) {
    return <div>Loading...</div>;
  }

  const transcript = transcripts[currentIndex];

  return (
    <div className="container mt-5">
      <h1>Transcription Review</h1>
      <form onSubmit={handleSubmit}>
        <div className="card mt-3">
          <div className="card-body">
            <textarea
              className="form-control"
              name="transcript"
              rows="3"
              defaultValue={transcript.transcript}
            />
            <audio src={transcript.url} controls className="mt-3 mb-3" />
          </div>
        </div>
        <div className="mt-3">
          <button type="submit" className="btn btn-primary">
            Save Changes
          </button>
          <button
            type="button"
            className="btn btn-primary ms-2"
            onClick={() => navigate(-1)}
          >
            Previous
          </button>
          <button
            type="button"
            className="btn btn-primary ms-2"
            onClick={() => navigate(1)}
          >
            Next
          </button>
          <button
            type="button"
            className="btn btn-success ms-2"
            onClick={handleThumbsUp}
          >
            Thumbs Up
          </button>
          <button
            type="button"
            className="btn btn-danger ms-2"
            onClick={handleThumbsDown}
          >
            Thumbs Down
          </button>
        </div>
      </form>
    </div>
  );
}

export default App;
