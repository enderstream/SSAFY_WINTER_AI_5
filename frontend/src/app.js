import MarkdownIt from 'markdown-it'

const chatContainer = document.getElementById("chat-container");
const messageForm = document.getElementById("message-form");
const userInput = document.getElementById("user-input");

const BASE_URL = process.env.API_ENDPOINT;

// Create a message bubble
function createMessageBubble(content, sender = "user") {
  const wrapper = document.createElement("div");
  wrapper.classList.add("mb-6", "flex", "items-start", "space-x-3");
  
  // Markdown-it 초기화
  const md = new MarkdownIt();
  
  // Avatar
  const avatar = document.createElement("div");
  avatar.classList.add(
    "w-10",
    "h-10",
    "rounded-full",
    "flex-shrink-0",
    "flex",
    "items-center",
    "justify-center",
    "font-bold",
    "text-white"
  );

  if (sender === "assistant") {
    avatar.classList.add("bg-gradient-to-br", "from-green-400", "to-green-600");
    avatar.textContent = "A";
  } else {
    avatar.classList.add("bg-gradient-to-br", "from-blue-500", "to-blue-700");
    avatar.textContent = "U";
  }

  // Bubble
  const bubble = document.createElement("div");
  bubble.classList.add(
    "max-w-full",
    "md:max-w-2xl",
    "p-3",
    "rounded-lg",
    "whitespace-pre-wrap",
    "leading-relaxed",
    "shadow-sm"
  );

  if (sender === "assistant") {
    bubble.classList.add("bg-gray-200", "text-gray-900");
  } else {
    bubble.classList.add("bg-blue-600", "text-white");
  }

  // HTML을 DOM에 삽입
  const html = md.render(content);
  bubble.innerHTML = html;

  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);
  return wrapper;
}

// Scroll to bottom
function scrollToBottom() {
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Fetch assistant response from the backend
async function getAssistantResponse(userMessage) {
  const url = "https://ssafy-winter-ai-5-be-kdh-v1.fly.dev/chat";
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ message: userMessage }),
  });

  if (!response.ok) {
    throw new Error("Network response was not ok");
  }

  const data = await response.json();
  return data.reply;
}


// Handle form submission
messageForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = userInput.value.trim();
  if (!message) return;

  // User message
  chatContainer.appendChild(createMessageBubble(message, "user"));
  userInput.value = "";
  scrollToBottom();

  // Assistant response
  const response = await getAssistantResponse(message);
  chatContainer.appendChild(createMessageBubble(response, "assistant"));
  scrollToBottom();
});
