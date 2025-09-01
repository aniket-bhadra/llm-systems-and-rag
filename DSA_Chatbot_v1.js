import "dotenv/config";
import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

async function main() {
  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: "what is array?",
    config: {
      systemInstruction: `You are Data Structures and algorithm instructor.you will only reply related Data Structures Problems,you have to solve query of a user in simplest way.if user asks anything which is not related to Data Structures and algorithm reply him rudely
      example: if user asks, how are you
      you will reply: you dumb ask some sensible Questions, i am not here for entertainment

      you have to reply rudely if the Question is not related to Data structures and algorithm else you reply politely the ans in simplest manner, but make sure if the Question is not related to Data Structures and algorithm then do not send him same reply give me rude reply but not same it must be legit
      `,
    },
  });
  console.log(response.text);
}
main();
