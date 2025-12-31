import { useState } from "react";

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);

  const askQuestion = async () => {
    if (!question.trim()) return;

    setLoading(true);
    setAnswer("");
    setSources([]);

    try {
      const res = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: question,
          top_k: 3,
        }),
      });

      const data = await res.json();
      setAnswer(data.answer);
      setSources(data.sources || []);
    } catch (err) {
      console.error(err);
      setAnswer("Error talking to backend");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#f4f6fb",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        fontFamily: "Inter, system-ui, sans-serif",
        padding: 20,
      }}
    >
      <div
        style={{
          width: "100%",
          maxWidth: 800,
          background: "#ffffff",
          padding: 24,
          borderRadius: 12,
          boxShadow: "0 10px 30px rgba(0,0,0,0.08)",
        }}
      >
        <h2 style={{ textAlign: "center", marginBottom: 20 }}>
          🧠 Local RAG Chat
        </h2>

        {/* Question box */}
        <textarea
          rows={3}
          placeholder="Ask a question..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          style={{
            width: "100%",
            padding: 12,
            borderRadius: 8,
            border: "1px solid #ccc",
            fontSize: 14,
            resize: "none",
          }}
        />

        {/* Ask button */}
        <button
          onClick={askQuestion}
          disabled={loading}
          style={{
            marginTop: 12,
            width: "100%",
            padding: "10px 16px",
            borderRadius: 8,
            border: "none",
            background: "#4f46e5",
            color: "#fff",
            fontSize: 15,
            cursor: "pointer",
          }}
        >
          {loading ? "Thinking..." : "Ask"}
        </button>

        {/* Answer */}
        {answer && (
          <div style={{ marginTop: 24 }}>
            <h3 style={{ marginBottom: 8 }}>Answer</h3>
            <div
              style={{
                background: "#f9fafb",
                padding: 14,
                borderRadius: 8,
                lineHeight: 1.6,
              }}
            >
              {answer}
            </div>
          </div>
        )}

        {/* Sources */}
        {sources.length > 0 && (
          <div style={{ marginTop: 20 }}>
            <h4 style={{ marginBottom: 8 }}>Sources</h4>
            <ul style={{ paddingLeft: 18 }}>
              {sources.map((s, i) => {
                // If backend returns string sources
                if (typeof s === "string") {
                  return (
                    <li key={i} style={{ marginBottom: 6 }}>
                      <small>{s}</small>
                    </li>
                  );
                }

                // If backend returns {chunk, similarity}
                return (
                  <li key={i} style={{ marginBottom: 6 }}>
                    <small>
                      ({typeof s.similarity === "number" ? s.similarity.toFixed(3) : "N/A"}){" "}
                      {s.chunk}
                    </small>
                  </li>
                );
              })}
            </ul>
          </div>
        )}
      </div>
    </div>
  );

}

export default App;
