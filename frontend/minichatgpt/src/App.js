import React, { useState } from 'react';
import axios from 'axios';
import './App.css';  // Ensure you have some basic styling

function App() {
  const [prompt, setPrompt] = useState('');
  const [conversation, setConversation] = useState([]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    const newConversation = [...conversation, { sender: 'user', text: `prompt: ${prompt}` }];
    setConversation(newConversation);
    setPrompt('');

    try {
      const result = await axios.get('http://localhost:5000/response', {
        params: { prompt }
      });
      setConversation([...newConversation, { sender: 'bot', text: `response: ${result.data.response}` }]);
    } catch (error) {
      console.error('Error fetching response:', error);
      setConversation([...newConversation, { sender: 'bot', text: 'response: Error fetching response' }]);
    }
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
          />
          <button type="submit" className="chat-button">Send</button>
        </form>
      </header>
    </div>
  );
}

export default App;
