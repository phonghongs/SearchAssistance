import React, { useState } from "react";
import { ChatInput } from "./ChatInput";
import { ChatHistory } from "./ChatHistory";

export interface Message {
  content: string;
  role: string;
}

export const ChatClient: React.FC = () => {
  const [messages, setMessages] = useState<Array<Message>>([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleSimpleChat = (message: string) => {
    // Send the message and past chat history to the backend
    // Update messages state with the new message
    let newMessages = [...messages, { content: message, role: "user" }];
    setMessages(newMessages);
    let postData = {
      message: newMessages,
    };
    setIsLoading(true);
    fetch("/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(postData),
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.status === "success") {
          setMessages([
            ...newMessages,
            { content: data.message, role: "assistant" },
          ]);
        }
        setIsLoading(false);
        console.log("Success:", data);
      })
      .catch((error) => {
        console.error("Error:", error);
        setIsLoading(false);
      });
  };

  const handleAdvancedChat = (message: string) => {
    // Trigger AI agent with Google Search functionality
    // Update messages state with the new message and AI response
    let newMessages = [...messages, { content: message, role: "user" }];
    setMessages(newMessages);
    let postData = {
      message: newMessages,
    };
    setIsLoading(true);
    fetch("/search", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(postData),
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.status === "success") {
          setMessages([
            ...newMessages,
            { content: data.message, role: "assistant" },
          ]);
        }
        console.log("Success:", data);
        setIsLoading(false);
      })
      .catch((error) => {
        console.error("Error:", error);
        setIsLoading(false);
      });
  };

  return (
    <div className="h-screen bg-gray-100 dark:bg-gray-900 flex items-center justify-center">
      <div className="flex flex-col items-center gap-2">
        <h1 className="text-white text-xl">
          AI Assistant with Claude and LangChain
        </h1>
        <div className="w-full max-w-md h-[80vh] bg-white dark:bg-gray-800 rounded-lg shadow-md overflow-hidden flex flex-col">
          <ChatHistory messages={messages} isLoading={isLoading} />
          <ChatInput
            onSimpleChat={handleSimpleChat}
            onAdvancedChat={handleAdvancedChat}
          />
        </div>
      </div>
    </div>
  );
};
