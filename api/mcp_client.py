from ast import arg
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import traceback

from utils.logger import logger

from anthropic import Anthropic
from anthropic.types import Message

import os

from datetime import datetime
import json

class MCPClient:    
    ## Init function
    def __init__(self):
        self.session = None
        self.exit_stack = AsyncExitStack()
        self.llm = Anthropic()
        self.tools = []
        self.messages = []
        self.logger = logger


    ## Connect to MCP server '
    async def connect_to_server(self, server_script_path:str):
        try:
            is_python = server_script_path.endswith(".py")

            if not is_python:
                raise ValueError("Server script must be a python file.")

            command = "python"
            server_params = StdioServerParameters(
                command=command,
                args=[server_script_path],
                env=None
            )

            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )

            await self.session.initialize()

            self.logger.info("Connected to MCP.")

            mcp_tools = await self.get_mcp_tools()
            
            self.tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
                for tool in mcp_tools
            ]

            self.logger.info(f"Available Tools: {[tool['name'] for tool in self.tools]}")


            return True


        except Exception as e:
            self.logger.error(f"Error connecting to MCP server: {e}")
            traceback.print_exc()
            raise


    ## Get Tools List
    async def get_mcp_tools(self):
        try:
            response = await self.session.list_tools()
            return response.tools
        except Exception as e:
            self.logger.error(f"Error getting MCP tools: {e}")
    

    ## Process Query
    async def process_query(self, query:str):
        try:
            self.logger.info(f"Processing Query: {query}")
            user_meesage = {"role": "user", "content": query}
            self.messages = [user_meesage]

            while True:
                response = await self.call_llm()

                ## text message
                if response.content[0].type == 'text' and len(response.content) == 1:
                    assistant_message = {
                        "role": "assistant",
                        "content": response.content[0].text
                    }
                    self.messages.append(assistant_message)
                    await self.log_conversation()
                    break

                ## not text message (e.g. tool call)
                assistant_message = {
                    "role": "assistant",
                    "content": response.to_dict()['content'],
                }
                self.messages.append(assistant_message)
                await self.log_conversation()

                for content in response.content:
                    if content.type == "tool_use":
                        tool_name=content.name
                        tool_args=content.input
                        tool_use_id=content.id

                        self.logger.info(
                            f"Calling Tool {tool_name} with args {tool_args}"
                        )

                        try:
                            result = await self.session.call_tool(tool_name, tool_args)
                            self.logger.info(f"Tool {tool_name} Result: {result}...")
                            self.messages.append(
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "tool_result",
                                            "tool_use_id": tool_use_id,
                                            "content": result.content,
                                        }
                                    ]
                                }
                            )
                        
                            await self.log_conversation()

                        except Exception as e:
                            self.logger.error(f"Error calling tool {tool_name}: {e}")
                            raise

            return self.messages
        
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            raise


    ## Call LLM
    async def call_llm(self):
        try:
            self.logger.info("Calling LLM.")
            return self.llm.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=1000,
                messages=self.messages,
                tools=self.tools,
            )
        except Exception as e:
            self.logger.error("Error calling LLM.")
            raise


    ## Clean up 
    async def cleanup(self):
        try:
            await self.exit_stack.aclose()
            self.logger.info("Disconnected From MCP Server.")
        except Exception as e:
            self.logger.error(f"Error during cleanup.")
            raise


    ## Log Conversation
    async def log_conversation(self):
        os.makedirs("conversations", exist_ok=True)

        serializable_conversation = []

        for message in self.messages:
            try:
                serializable_message = {"role": message["role"], "content": []}

                # Handle both string and list content
                if isinstance(message["content"], str):
                    serializable_message["content"] = message["content"]
                elif isinstance(message["content"], list):
                    for content_item in message["content"]:
                        if hasattr(content_item, "to_dict"):
                            serializable_message["content"].append(
                                content_item.to_dict()
                            )
                        elif hasattr(content_item, "dict"):
                            serializable_message["content"].append(content_item.dict())
                        elif hasattr(content_item, "model_dump"):
                            serializable_message["content"].append(
                                content_item.model_dump()
                            )
                        else:
                            serializable_message["content"].append(content_item)

                serializable_conversation.append(serializable_message)
            except Exception as e:
                self.logger.error(f"Error processing message: {str(e)}")
                self.logger.debug(f"Message content: {message}")
                raise

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filepath = os.path.join("conversations", f"conversation_{timestamp}.json")

        try:
            with open(filepath, "w") as f:
                json.dump(serializable_conversation, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Error writing conversation to file: {str(e)}")
            self.logger.debug(f"Serializable conversation: {serializable_conversation}")
            raise


    
