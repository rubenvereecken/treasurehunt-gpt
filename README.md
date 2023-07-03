# Treasure Hunt GPT

Treasure Hunt Admin Bot, written on top of [Langchain](https://github.com/hwchase17/langchain) and GPT 3.5, for a Treasure Hunt that I organised. See [this notebook](<Treasure Hunt.ipynb>) to see the bot in action, including the prompts used to direct it.

Frontend (with screenshots) lives [here](https://github.com/rubenvereecken/treasurehunt-frontend)

## State Machine

Instead of relying on the bot to _remember_ which state (treasure hunt riddle) it should be on, it is instead forced to simulate a state machine. In each state, it should figure out the best action to take. The rules of the state machine are explained in simple natural language, as opposed to a formal representation.

(TODO: add screenshot for the state machine governing the bot)

## Tech Stack

- Python 3.9
- Langchain and GPT 3.5 (optimised for chat)
- DynamoDB for chat history
- Serverless Framework + AWS Lambda/API Gateway
