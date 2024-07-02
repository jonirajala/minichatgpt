import React, { useState } from 'react';
import './App.css';

function App() {
  const [prompt, setPrompt] = useState('');
  const [conversation, setConversation] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    const newConversation = [...conversation, { sender: 'user', text: `prompt: ${prompt}` }];
    setConversation(newConversation);
    setPrompt('');
    setLoading(true);

    const eventSource = new EventSource(`http://localhost:5000/response-stream?prompt=${encodeURIComponent(prompt)}`);

    eventSource.onmessage = (event) => {
      setConversation((prevConversation) => {
        const lastMessage = prevConversation[prevConversation.length - 1];
        if (lastMessage && lastMessage.sender === 'bot') {
          return [
            ...prevConversation.slice(0, -1),
            { ...lastMessage, text: lastMessage.text + event.data },
          ];
        }
        return [...prevConversation, { sender: 'bot', text: 'answer: ' + event.data }];
      });
    };

    eventSource.onerror = (error) => {
      console.error('Error fetching response:', error);
      // setConversation((prevConversation) => [...prevConversation, { sender: 'bot', text: 'response: Error fetching response' }]);
      eventSource.close();
      setLoading(false);
    };

    eventSource.onopen = () => {
      setLoading(false);
    };

    eventSource.onend = () => {
      setLoading(false);
      eventSource.close();
    };

    // This ensures we close the event source after receiving all messages
    eventSource.addEventListener('end', () => {
      setLoading(false);
      eventSource.close();
    });
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Chat with Model</h1>
        <div className="chat-container">
          {conversation.map((msg, index) => (
            <div key={index} className={`chat-message ${msg.sender}`}>
              <div className="message-text">{msg.text}</div>
            </div>
          ))}
        </div>
        <form onSubmit={handleSubmit} className="chat-form">
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Enter your prompt"
            className="chat-input"
            disabled={loading}
          />
          <button type="submit" className="chat-button" disabled={loading}>
            {loading ? 'Loading...' : 'Send'}
          </button>
        </form>
      </header>
    </div>
  );
}

export default App;
