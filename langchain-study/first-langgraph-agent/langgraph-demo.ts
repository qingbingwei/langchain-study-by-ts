// define tools and model
import { tool } from "@langchain/core/tools";
import { ChatDeepSeek } from "@langchain/deepseek";
import { appConfig } from "../../config/config";
import * as z from "zod";

// define the state graph
import { StateGraph, START, END, messagesStateReducer } from "@langchain/langgraph";
import { MessagesZodMeta } from "@langchain/langgraph";
import { registry } from "@langchain/langgraph/zod";
import { type BaseMessage } from "@langchain/core/messages";

// define model node
import { SystemMessage } from "@langchain/core/messages";

// define the tool node
import { AIMessage, ToolMessage } from "@langchain/core/messages";

// Invoke the agent
import { HumanMessage } from "@langchain/core/messages";

// enable langsmith tracing
import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from 'url';

// 修复 __dirname 在 ES Module 中的使用
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// 无论在哪个目录运行脚本，都能加载脚本同目录的 .env
dotenv.config({ path: path.resolve(__dirname, ".env") });

// define the deepseek chat model
const deepseekChatModel = new ChatDeepSeek({
  model: appConfig.deepseekModel,
  apiKey: appConfig.deepseekApiKey,
  temperature: 0.5,
});

// define the tools
const add = tool(
    (input) => input.a + input.b,
    {
        name: "add",
        description: "Add two numbers together",
        schema: z.object({
            a: z.number().describe("The first number"),
            b: z.number().describe("The second number"),
        }),
    }
);

const sub = tool(
    (input) => input.a - input.b,
    {
        name: "subtract",
        description: "Subtract two numbers together",
        schema: z.object({
            a: z.number().describe("The first number"),
            b: z.number().describe("The second number"),
        }),
    }
);

const mul = tool(
    (input) => input.a * input.b,
    {
        name: "multiply",
        description: "Multiply two numbers together",
        schema: z.object({
            a: z.number().describe("The first number"),
            b: z.number().describe("The second number"),
        }),
    }
);

const div = tool(
    (input) => input.a / input.b,
    {
        name: "divide",
        description: "Divide two numbers together",
        schema: z.object({
            a: z.number().describe("The first number"),
            b: z.number().describe("The second number"),
        }),
    }
);

const toolsByName = {
  [add.name]: add,
  [sub.name]: sub,
  [mul.name]: mul,
  [div.name]: div,
};
const tools = Object.values(toolsByName);
const modelWithTools = deepseekChatModel.bindTools(tools);

// define the state
const MessagesState = z.object({
  messages: z
    .array(z.custom<BaseMessage>())
    .register(registry, MessagesZodMeta),
  llmCalls: z.number().optional(),
});

// define the llm node
async function llmCall(state: z.infer<typeof MessagesState>) {
    return {
        messages: await modelWithTools.invoke([
            new SystemMessage({
                content: "You are a helpful assistant tasked with performing arithmetic on a set of inputs.",
            }),
            ...state.messages,
        ]),
        llmCalls: (state.llmCalls ?? 0) + 1,
    };
}

// define the tool node
async function toolNode(state: z.infer<typeof MessagesState>) {
    const lastMessage = state.messages.at(-1);

    if (lastMessage == null || !(lastMessage instanceof AIMessage)) {
        return { messages:[] };
    }

    const result: ToolMessage[] = [];
    for (const toolCall of lastMessage.tool_calls ?? []) {
        const cur_tool = toolsByName[toolCall.name];
        if (cur_tool == null) {
            continue;
        }
        const observation = await cur_tool.invoke(toolCall);
        result.push(observation)
    }

    return { messages: result };
}

// define the end logic
function shouldContinue(state: z.infer<typeof MessagesState>) {
    const lastMessage = state.messages.at(-1);
    if (lastMessage == null || !(lastMessage instanceof AIMessage)) {
        return END;
    }
    if(lastMessage.tool_calls?.length) {
        return "toolNode";
    }

    // Otherwise, we stop (reply to the user)
    return END;
}

// build and compile the agent
const agent = new StateGraph(MessagesState)
    .addNode("llmCall",llmCall)
    .addNode("toolNode",toolNode)
    .addEdge(START,"llmCall")
    .addConditionalEdges("llmCall",shouldContinue,["toolNode",END])
    .addEdge("toolNode","llmCall")
    .compile();

// Invoke the agent
const result = await agent.invoke({
    messages: [new HumanMessage({ content: "Add 1 and 2" })],
});

for (const message of result.messages) {
    console.log(`[${message.type}]: ${message.text}`);
}