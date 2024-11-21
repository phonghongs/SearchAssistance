import React from "react";
import ReactMarkdown from "react-markdown";
import { Message } from "./ChatClient";

interface ChatHistoryProps {
  messages: Array<Message>;
  isLoading: boolean;
}

export const ChatHistory: React.FC<ChatHistoryProps> = ({
  messages,
  isLoading,
}) => {
  return (
    <div className="p-4 h-full overflow-y-auto">
      {messages.map((message, index) => (
        <div
          key={index}
          className={`mb-3 ${
            message.role === "user" ? "text-right" : "text-left"
          }`}
        >
          <ReactMarkdown
            className={`inline-block px-3 py-2 rounded-md ${
              index % 2 === 0
                ? "bg-gray-300 dark:bg-gray-700"
                : "bg-blue-200 dark:bg-blue-900"
            }`}
          >
            {message.content}
          </ReactMarkdown>
        </div>
      ))}
      {isLoading && (
        <div className="mb-3 text-left">
          <ReactMarkdown className="inline-block px-3 py-2 rounded-md bg-blue-200 dark:bg-blue-900 animate-pulse">
            {/* Put your desired loading content here */}
            Thinking...
          </ReactMarkdown>
        </div>
      )}
    </div>
  );
};
