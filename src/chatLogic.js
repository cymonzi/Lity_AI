// chatLogic.js
export async function chatWithBot(userMessage) {
  try {
    const response = await fetch("https://f52896ca977b.ngrok-free.app/chat/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text: userMessage }),
    });
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API error: ${response.status} - ${errorText}`);
    }
    const data = await response.json();
    return data.reply;
  } catch (error) {
    console.error("Chat API error:", error);
    return `Sorry, something went wrong. (${error.message})`;
  }
}

export async function checkBackendHealth() {
  try {
    const response = await fetch("https://f52896ca977b.ngrok-free.app/");
    return response.ok;
  } catch {
    return false;
  }
}
