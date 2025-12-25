import { createAgent,tool,type ToolRuntime } from "langchain";
import { ChatDeepSeek } from "@langchain/deepseek";
import { appConfig } from "../../config/config";
import { MemorySaver } from "@langchain/langgraph";

import * as z from "zod";

const systemPrompt = `You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location.`;

const getWeather = tool(
    (input) => `It is sunny in ${input.city}!`,
    {
        name: "get_weather_for_location",
        description: "Get the weather for a given city",
        schema: z.object({
            city: z.string().describe("The city to get the weather for"),
        }),
    }
)

//
type AgentRuntime = ToolRuntime<unknown, { user_id: string }>;

const getUserLocation = tool(
  (_,config: AgentRuntime) => {
    const {user_id} = config.context
    return user_id === "1" ? "Tokyo" : "Shanghai"
  },
  {
    name: "get_user_location",
    description: "Get the user's location by user_id",
  }
);

// Add a checkpointer to the agent
const checkpointer = new MemorySaver();

// Add a response format to the agent
const responseFormat = z.object({
  punny_response: z.string(),
  weather_conditions: z.string().optional(),
});

const deepseekChatModel = new ChatDeepSeek({
  model: appConfig.deepseekModel,
  apiKey: appConfig.deepseekApiKey,
  temperature: 0.5,
})

const agent = createAgent({
    model: deepseekChatModel,
    tools: [getWeather,getUserLocation],
    systemPrompt: systemPrompt,
    responseFormat: responseFormat,
    checkpointer: checkpointer,
})

const config = {
  configurable: {
    thread_id: "1",
  },
  context: {
    user_id: "1",
  }
}
const response = await agent.invoke(
  {
    messages: [{ role: "user", content: "What's the weather outside?" }]
  },
  config,
)
console.log(response.structuredResponse);

const thankYouResponse = await agent.invoke(
  {
    messages: [{ role: "user", content: "Thanks for your weather report!" }]
  },
  config,
)

console.log(thankYouResponse.structuredResponse);
