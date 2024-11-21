import React, { useState } from "react";

interface ChatInputProps {
  onSimpleChat: (message: string) => void;
  onAdvancedChat: (message: string) => void;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSimpleChat,
  onAdvancedChat,
}) => {
  const [input, setInput] = useState("");

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setInput(event.target.value);
  };

  const handleSubmit = (handler: (message: string) => void) => {
    handler(input);
    setInput("");
  };

  return (
    <div className="flex p-4 border-t border-gray-200 dark:border-gray-700">
      <input
        type="text"
        value={input}
        onChange={handleInputChange}
        placeholder="Type your message..."
        className="flex-grow px-3 py-2 rounded-md bg-gray-200 text-gray-900 dark:bg-gray-700 dark:text-gray-100 focus:outline-none"
      />
      <button
        onClick={() => handleSubmit(onSimpleChat)}
        className="ml-2 px-4 py-2 font-semibold text-gray-600 bg-white dark:text-gray-400 dark:bg-gray-800 border border-gray-300 rounded-md hover:bg-gray-200 dark:hover:bg-gray-700 focus:outline-none"
      >
        Ask
      </button>
      <button
        onClick={() => handleSubmit(onAdvancedChat)}
        className="ml-2 px-4 py-2 font-semibold text-white bg-blue-500 border border-blue-600 rounded-md hover:bg-blue-400 focus:outline-none"
      >
        Ask and Search
      </button>
    </div>
  );
};
