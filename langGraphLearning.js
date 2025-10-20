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
const toolsByName = Object.fromEntries(
  availableTools.map((tool) => [tool.name, tool])
);

// => [['add', add], ['Multiply', multiply], ['divide', divide]]
// {
//   add: add,
//   Multiply: multiply,
//   divide: divide
// }

const llmWithTools = llm.bindTools(availableTools);

// What is state?

// state is an object that LangGraph automatically passes to all the functions
// It contains: state.messages which is an array of all conversation messages

// state = {
//   messages: [
//     { role: "user", content: "Add 3 and 4" }
//   ]
// }

// State is an object that is passed by LangGraph to every function in the agent workflow, but what that state contains - that we can decide. For example, here we did new StateGraph(MessagesAnnotation), so it tells state will have a messages attribute which is an array. This way we could have used any other configuration which tells state will contain attributes like fname, lname, email, etc.

// Key point: MessagesAnnotation is a predefined schema from LangGraph specifically for message-based conversations. For custom attributes, you'd define your own state schema!

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

// What is ToolNode?

// It's a ready-made node that LangGraph provides
// Its job: Execute the tools that the LLM asks for

// How does it work?
// When the LLM says "I want to use the add tool with a=3, b=4":

// ToolNode looks at the tool call
// Finds the `add` tool from availableTools
// Executes it: add(3, 4)
// Gets the result: 7
// Creates a message with this result
// Adds this message to state.messages

// Ex --
// {
//   role: "tool",
//   content: "7",  // The result of add(3, 4)
//   name: "add"
// }

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


// shouldContinue doesn't execute the toolNode - it just tells the graph WHERE to go next!

// Here's how it works:

// shouldContinue returns a string: "toolNode" or "__end__"
// LangGraph reads that string and says "okay, go to the node with that name"
// LangGraph then executes the actual toolNode
// toolNode runs, executes the tool, creates a message, and adds it to state
// Build workflow


const agentBuilder = new StateGraph(MessagesAnnotation)
// Creates a new graph
// MessagesAnnotation means: "The state will have a messages property that's an array

   // Add nodes 
  .addNode("llmCall", llmCall) // Add a node called "llmCall" that runs the llmCall function
  .addNode("toolNode", toolNode) // Add a node called "toolNode" that runs the toolNode

  // Add edges to connect nodes
  .addEdge("__start__", "llmCall") // When we start, always go to "llmCall" first
  .addConditionalEdges("llmCall", shouldContinue, ["toolNode", "__end__"]) 
      //   From: "llmCall" node
      // Use: shouldContinue function to decide where to go --> toolNode or __end__
      // 3rd argument = List of possible destination nodes shouldContinue can return,It tells LangGraph: "These are the ONLY valid places shouldContinue can route to. if it returns something else (e.g., "randomNode") → ERROR! ❌ (not in the list)
  .addEdge("toolNode", "llmCall") // After executing tools, go back to llmCall
  .compile(); // Build the final graph



//   If returns "toolNode":
// Graph goes to toolNode → executes tools → then follows the edge toolNode → llmCall (defined in .addEdge("toolNode", "llmCall"))

// If returns "__end__":
// Graph stops, conversation ends, final state.messages is returned to user as output

// Invoke
const messages = [
  {
    role: "user",
    content: "Add 3 and 4.",
  },
];
const result = await agentBuilder.invoke({ messages });
// We invoke the graph with this initial state,so, This is the initial state of state.messages when you invoke the graph.
console.log(result.messages);


// ## **The Complete Flow (Visual)**
// ```
// START
//   ↓
// llmCall (LLM thinks)
//   ↓
// shouldContinue() checks: "Did LLM request tools?"
//   ↓                    ↓
//   YES                  NO
//   ↓                    ↓
// toolNode            __END__
// (execute tools)
//   ↓
//   ↓ (loop back)
// llmCall (LLM sees tool results)
//   ↓
// shouldContinue() checks again
//   ↓
//   NO (this time)
//   ↓
// __END__


// ### **Execution Trace:**

// **🔵 Round 1:**
// 1. **START → llmCall**
//    - State: `messages = [{ role: "user", content: "Add 3 and 4." }]`
//    - LLM receives: system message + user message
//    - LLM thinks: "I need to use add tool"
//    - LLM returns: `{ role: "assistant", tool_calls: [{ name: "add", args: { a: 3, b: 4 } }] }`
//    - This gets added to state.messages

// 2. **llmCall → shouldContinue**
//    - Checks last message
//    - Sees `tool_calls` exists
//    - Returns `"toolNode"`

// 3. **Goes to toolNode**
//    - Executes `add(3, 4)` → gets `7`
//    - Adds: `{ role: "tool", content: "7" }`
//    - Now state.messages has: user message, assistant tool call, tool result

// 4. **toolNode → llmCall** (loop back)

// **🔵 Round 2:**
// 5. **llmCall again**
//    - State now has: user message, AI tool call, tool result (7)
//    - LLM sees the tool returned 7
//    - LLM thinks: "Great! I have the answer"
//    - LLM returns: `{ role: "assistant", content: "The answer is 7." }`
//    - No `tool_calls` this time!

// 6. **llmCall → shouldContinue**
//    - Checks last message
//    - No `tool_calls` found
//    - Returns `"__end__"`

// 7. **Goes to END**

// **Step 4:** Final result:
// ```javascript
// result.messages = [
//   { role: "user", content: "Add 3 and 4." },
//   { role: "assistant", tool_calls: [...] },
//   { role: "tool", content: "7" },
//   { role: "assistant", content: "The answer is 7." }
// ]
// ```


// role: "user" → Messages from the human user (queries, requests)
// role: "assistant" → Messages from the AI/LLM (responses, tool calls)
// role: "system" → Instructions/context for the AI (tells it how to behave)
// role: "tool" → Results returned by tools after execution (e.g., "7" from add function)

// So: role: "tool" = convention for tool results, not specific to ToolNode. So whether we use the prebuilt ToolNode or we manually do tool execution, for the results of tools we always put role: "tool"


//imp Qs

// ## **1. Is `shouldContinue` a node?**

// **NO! `shouldContinue` is NOT a node.**

// ### **What is it then?**

// `shouldContinue` is a **routing function** (also called a **conditional edge function** or **decision function**).

// ### **Difference:**

// - **Nodes** = Do work (execute logic, call LLM, run tools, modify state)
//   - `llmCall` → calls the LLM
//   - `toolNode` → executes tools

// - **Routing functions** = Make decisions (just return a string to say where to go next, don't modify state)
//   - `shouldContinue` → returns `"toolNode"` or `"__end__"`

// ### **Why not add it as a node?**

// Because it doesn't DO anything - it just DECIDES where to go!

// **Think of it like:**
// - **Nodes** = Workers doing tasks 👷
// - **Routing functions** = Traffic signs pointing directions 🚦

// ---

// ## **2. What does `.compile()` do?**

// **It builds the final workflow.**

// ### **What compile does:**

// ```javascript
// .compile();  
// ```

// 1. **Takes all the nodes and edges** you defined
// 2. **Validates** the graph structure (checks for errors)
// 3. **Builds the executable workflow**
// 4. **Returns a runnable graph** that you can invoke

// ### **Before compile:**
// ```
// agentBuilder = [just a blueprint/recipe, can't run yet]
// ```

// ### **After compile:**
// ```
// agentBuilder = [executable workflow ready to run!]
// ```
