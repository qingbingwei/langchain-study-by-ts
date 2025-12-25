import { createAgent,tool } from "langchain";
import { ChatDeepSeek } from "@langchain/deepseek";
import { appConfig } from "../../config/config";

import * as z from "zod";

const getWeather = tool(
    (input) => `It is sunny in ${input.city}!`,
    {
        name: "get_weather",
        description: "Get the weather for a given city",
        schema: z.object({
            city: z.string().describe("The city to get the weather for"),
        }),
    }
)

const deepseekChatModel = new ChatDeepSeek({
  model: "deepseek-chat",
  apiKey: appConfig.deepseekApiKey,
})

const agent = createAgent({
    model: deepseekChatModel,
    tools: [getWeather],
})

console.log(
  await agent.invoke({
    messages: [{ role: "user", content: "What's the weather in Tokyo?" }],
  })
);
