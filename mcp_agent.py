import textarena as ta
from textarena.agents.basic_agents import AsyncAnthropicAgent
import smithery
import mcp
import os
import json


class MCPAgent(AsyncAnthropicAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.url = smithery.create_smithery_url(
            "wss://server.smithery.ai/exa/ws", {"exaApiKey": os.environ["EXA_API_KEY"]}
        )

    async def _make_request(self, observation: str) -> str:
        """Make a single API request to Anthropic and return the generated message."""
        async with smithery.websocket_client(self.url) as streams:
            async with mcp.client.session.ClientSession(*streams) as session:

                try:
                    tools_result = await session.list_tools()
                    tools = tools_result.model_dump()["tools"]

                    tools = [
                        {"input_schema": tool.pop("inputSchema"), **tool}
                        for tool in tools
                        if "inputSchema" in tool
                    ]

                    print("Available tools:", tools)

                    final_response_text = ""
                    is_tool_call_pending = True
                    messages = [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": observation}],
                        }
                    ]

                    # Loop to handle multiple tool calls in a conversation
                    while is_tool_call_pending:
                        response = await self.client.messages.create(
                            model=self.model_name,
                            max_tokens=self.max_tokens,
                            temperature=self.temperature,
                            system=self.system_prompt,
                            messages=messages,
                            tools=tools,
                        )

                        print("Response:", response)

                        # Check if there's a tool_use in the response
                        is_tool_call_pending = False
                        for content_block in response.content:
                            if content_block.type == "tool_use":
                                is_tool_call_pending = True

                                tool_name = content_block.name
                                tool_input = content_block.input
                                tool_id = content_block.id

                                print(f"Tool called: {tool_name}")
                                print(f"Tool input: {json.dumps(tool_input, indent=2)}")

                                # Execute the tool using MCP session
                                try:
                                    tool_result = await session.call_tool(
                                        tool_name, tool_input
                                    )

                                    # Convert tool result to string format for Anthropic
                                    # The content must be a string, not an object
                                    tool_result_dict = tool_result.model_dump()
                                except Exception as e:
                                    if "MCP error" in str(e):
                                        tool_result_dict = {"error": str(e)}

                                result_str = json.dumps(tool_result_dict)[:20000]
                                print(f"Tool result: {result_str}")

                                # Add tool call and result to messages
                                messages.append(
                                    {
                                        "role": "assistant",
                                        "content": [content_block.model_dump()],
                                    }
                                )

                                # Add tool response to messages - content must be a string
                                messages.append(
                                    {
                                        "role": "user",
                                        "content": [
                                            {
                                                "type": "tool_result",
                                                "tool_use_id": tool_id,
                                                "content": result_str,  # Now it's a string
                                            }
                                        ],
                                    }
                                )
                            elif content_block.type == "text":
                                # Accumulate text responses
                                final_response_text += content_block.text

                        # If no tool calls were made, we use the text response
                        if not is_tool_call_pending and not final_response_text:
                            final_response_text = response.content[0].text

                except Exception as e:

                    print(f"Error: {e}")
                    raise e

            return final_response_text.strip()
