import { config } from "dotenv";
import { tool } from "@langchain/core/tools";
import * as z from "zod";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { MessagesAnnotation, StateGraph } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { SystemMessage, ToolMessage } from "@langchain/core/messages";

config();

const llm = new ChatGoogleGenerativeAI({
  model: "gemini-2.5-flash",
  apiKey: process.env.GEMINI_API_KEY,
});

const multiply = tool(
  async ({ a, b }) => {
    return a * b;
  },
  {
    name: "Multiply",
    description: "Multiply two numbers",
    schema: z.object({
      a: z.number().describe("first number"),
      b: z.number().describe("second number"),
    }),
  }
);

const add = tool(
  ({ a, b }) => {
    return a + b;
  },
  {
    name: "add",
    description: "Add two numbers together",
    schema: z.object({
      a: z.number().describe("first number"),
      b: z.number().describe("second number"),
    }),
  }
);

const divide = tool(
  ({ a, b }) => {
    return a / b;
  },
  {
    name: "divide",
    description: "Divide two numbers",
    schema: z.object({
      a: z.number().describe("first number"),
      b: z.number().describe("second number"),
    }),
  }
);

const availableTools = [add, multiply, divide];


const llmWithTools = llm.bindTools(availableTools);


async function llmCall(state) {
  // LLM decides whether to call a tool or not
  const result = await llmWithTools.invoke([
    {
      role: "system",
      content:
        "You are a helpful assistant tasked with performing arithmetic on a set of inputs.",
    }, // system instructions
    ...state.messages, // All previous messages (user messages, AI responses, tool results)
  ]);

  return {
    messages: [result],
    //The LLM responds with result =>  ans is 7 or request an tool, this gets added to msgs array. When we return { messages: [result] }, LangGraph appends this to the existing state.messages. It doesn't replace them!
  };
}

const toolNode = new ToolNode(availableTools);


// Conditional edge function to route to the tool node or end
function shouldContinue(state) {
  const messages = state.messages;
  const lastMessage = messages.at(-1);

  // When the LLM wants to use a tool, it adds tool_calls to its response:
  if (lastMessage?.tool_calls?.length) {
    return "toolNode";
  }
  // Otherwise, we stop (reply to the user)
  return "__end__";
}

const agentBuilder = new StateGraph(MessagesAnnotation)
// Creates a new graph
// MessagesAnnotation means: "The state will have a messages property that's an array

   // Add nodes 
  .addNode("llmCall", llmCall) // Add a node called "llmCall" that runs the llmCall function
  .addNode("toolNode", toolNode) // Add a node called "toolNode" that runs the toolNode

  // Add edges to connect nodes
  .addEdge("__start__", "llmCall") // When we start, always go to "llmCall" first
  .addConditionalEdges("llmCall", shouldContinue, ["toolNode", "__end__"]) 
  .addEdge("toolNode", "llmCall") // After executing tools, go back to llmCall
  .compile(); // Build the final graph

// Invoke
const messages = [
  {
    role: "user",
    content: "Add 3 and 4 then multiply that by 10 and then divide the result by 2 then add 5",
  },
];
const result = await agentBuilder.invoke({ messages });
// We invoke the graph with this initial state,so, This is the initial state of state.messages when you invoke the graph.
console.log(result.messages);
